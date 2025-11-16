import io
import requests
import gradio as gr

BACKEND_URL = "http://127.0.0.1:8000/predict_cobb"

def cobb_client(image):
    if image is None:
        return "Please upload an X-ray image."

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"files": ("upload.png", buf, "image/png")}

    try:
        r = requests.post(BACKEND_URL, files=files)
        data = r.json()
        res = data["results"][0]
        thor = res["thoracic_cobb_deg"]
        lum  = res["lumbar_cobb_deg"]

        msg = (
            "🩻 **Cobb Angle AI Result**\n\n"
            f"- Thoracic Cobb: **{thor:.2f}°**\n"
            f"- Lumbar Cobb: **{lum:.2f}°**\n"
        )
        return msg
    except Exception as e:
        return f"Error calling backend: {e}"

demo = gr.Interface(
    fn=cobb_client,
    inputs=gr.Image(type="pil", label="Upload spine X-ray"),
    outputs=gr.Markdown(label="AI Output"),
    title="Cobb Angle AI",
    description="Upload a spine X-ray; the AI predicts thoracic & lumbar Cobb angles.",
)

if __name__ == "__main__":
    demo.launch()
