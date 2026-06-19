from __future__ import annotations

import os
from urllib.parse import quote

import requests
import streamlit as st

API_URL = os.getenv("ADAPTIVE_VISION_API_URL", "http://localhost:8000").rstrip("/")


def api_request(method: str, path: str, **kwargs) -> dict:
    timeout = kwargs.pop("timeout", 180)
    response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    if response.ok:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except requests.JSONDecodeError:
        detail = response.text
    raise RuntimeError(f"API {response.status_code}: {detail}")


def file_payload(uploaded_file) -> tuple[str, bytes, str]:
    return (
        uploaded_file.name,
        uploaded_file.getvalue(),
        uploaded_file.type or "application/octet-stream",
    )


st.set_page_config(page_title="Adaptive Vision", page_icon="AV", layout="wide")
st.title("Adaptive Vision")
st.caption("Teach visual classes from examples, classify new images, and review drift signals.")

with st.sidebar:
    st.subheader("Service")
    st.code(API_URL, language=None)
    try:
        health = api_request("GET", "/health", timeout=10)
        st.success("API online")
        st.caption(f"Model: {health['model']}")
        st.caption(f"Classes: {health['classes']}")
    except (requests.RequestException, RuntimeError) as exc:
        st.error("API unavailable")
        st.caption(str(exc))

predict_tab, teach_tab, monitor_tab = st.tabs(["Predict", "Teach", "Monitor"])

with predict_tab:
    left, right = st.columns([1, 1], gap="large")
    with left:
        predict_file = st.file_uploader(
            "Image",
            type=["png", "jpg", "jpeg", "webp"],
            key="predict_file",
        )
        top_k = st.number_input("Candidates", min_value=1, max_value=20, value=3)
        if predict_file:
            st.image(predict_file, use_container_width=True)
        predict_clicked = st.button(
            "Run prediction",
            type="primary",
            disabled=predict_file is None,
            use_container_width=True,
        )

    with right:
        if predict_clicked and predict_file:
            try:
                result = api_request(
                    "POST",
                    f"/v1/predict?top_k={top_k}",
                    files={"file": file_payload(predict_file)},
                )
                st.session_state["last_prediction"] = result
                st.session_state["last_image"] = file_payload(predict_file)
            except (requests.RequestException, RuntimeError) as exc:
                st.error(str(exc))

        result = st.session_state.get("last_prediction")
        if result:
            if result["is_unknown"]:
                st.warning("No class passed the configured similarity threshold.")
            else:
                st.success(result["label"])

            rows = [
                {
                    "class": item["label"],
                    "similarity": round(item["similarity"], 4),
                    "examples": item["examples"],
                }
                for item in result["matches"]
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

            st.divider()
            correction = st.text_input("Correct class", key="correction")
            if st.button("Submit feedback", disabled=not correction):
                try:
                    feedback = api_request(
                        "POST",
                        f"/v1/feedback/{quote(correction, safe='')}",
                        files={"file": st.session_state["last_image"]},
                    )
                    st.success(
                        f"Updated {feedback['label']} to {feedback['total_examples']} examples."
                    )
                except (requests.RequestException, RuntimeError) as exc:
                    st.error(str(exc))

with teach_tab:
    label = st.text_input("Class name", placeholder="damaged connector")
    examples = st.file_uploader(
        "Reference images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="teach_files",
    )
    if examples:
        preview_columns = st.columns(min(len(examples), 4))
        for index, example in enumerate(examples[:4]):
            preview_columns[index].image(example, use_container_width=True)

    if st.button(
        "Add examples",
        type="primary",
        disabled=not label or not examples,
        use_container_width=True,
    ):
        try:
            result = api_request(
                "POST",
                f"/v1/classes/{quote(label, safe='')}/examples",
                files=[("files", file_payload(example)) for example in examples],
            )
            st.success(
                f"Added {result['examples_added']} examples. "
                f"{result['label']} now has {result['total_examples']}."
            )
        except (requests.RequestException, RuntimeError) as exc:
            st.error(str(exc))

with monitor_tab:
    if st.button("Refresh", use_container_width=False):
        st.rerun()
    try:
        metrics = api_request("GET", "/v1/metrics", timeout=10)
        classes = api_request("GET", "/v1/classes", timeout=10)["classes"]
        first, second, third, fourth = st.columns(4)
        first.metric("Classes", metrics["class_count"])
        second.metric("Examples", metrics["example_count"])
        third.metric("Predictions", metrics["observations"])
        fourth.metric("Unknown rate", f"{metrics['unknown_rate']:.1%}")

        st.subheader("Prototype memory")
        st.dataframe(classes, use_container_width=True, hide_index=True)

        st.subheader("Rolling signal")
        st.json(
            {
                "window_size": metrics["window_size"],
                "mean_top_similarity": metrics["mean_top_similarity"],
                "p10_top_similarity": metrics["p10_top_similarity"],
                "last_observation_at": metrics["last_observation_at"],
            }
        )
    except (requests.RequestException, RuntimeError) as exc:
        st.error(str(exc))
