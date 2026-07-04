"""Main entry point for the car-recognition-maintenance-advisor app."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Render the app shell."""
    st.title("car-recognition-maintenance-advisor")
    st.write("Upload a car image to begin the prediction and maintenance report flow.")


if __name__ == "__main__":
    main()
