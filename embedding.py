import streamlit as st
import pdfplumber
import io
import base64
from PIL import Image as PILImage
from google.cloud import aiplatform
from vertexai.preview.vision_models import MultiModalEmbeddingModel
from vertexai.vision_models import Image

aiplatform.init(project="your-project-id", location="your-location") 
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")  

#PDF EXTRACTION
def extract_pdf_content(pdf_bytes):
    text_chunks = []
    image_data_store = []  

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages):
            
            text = page.extract_text()
            if text:
                text_chunks.append(text)

            for img_index, img in enumerate(page.images):
                try:
                    bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
                    cropped_img = page.within_bbox(bbox).to_image(resolution=150)
                    
                    img_byte_arr = io.BytesIO()
                    cropped_img.original.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()

                    pil_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()

                    image_data_store.append({
                        "page": page_number,
                        "bytes": img_bytes,
                        "base64": img_base64
                    })

                except Exception as e:
                    st.warning(f"⚠️ Image extraction failed on page {page_number}: {e}")

    return text_chunks, image_data_store

#EMBEDDING 
def get_embeddings_for_pdf(pdf_bytes):
    texts, images = extract_pdf_content(pdf_bytes)
    embeddings_results = []

    for text in texts:
        try:
            resp = model.get_embeddings(contextual_text=text)
            embeddings_results.append({
                'type': 'text',
                'content': text,
                'vector': resp.text_embedding
            })
        except Exception as e:
            st.error(f"❌ Text embedding failed: {e}")

    for img_data in images:
        try:
            image_obj = Image(image_bytes=img_data["bytes"])
            resp = model.get_embeddings(image=image_obj)
            embeddings_results.append({
                'type': 'image',
                'content': None,
                'vector': resp.image_embedding,
                'base64': img_data["base64"],
                'page': img_data["page"]
            })
        except Exception as e:
            st.error(f"❌ Image embedding failed: {e}")

    return embeddings_results

st.set_page_config(page_title="PDF Embedder", layout="wide")
st.title("📄 PDF to Vertex AI Embeddings (Text + Images)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.success("✅ File uploaded!")

    with st.spinner("Processing PDF and generating embeddings..."):
        pdf_bytes = uploaded_file.read()
        results = get_embeddings_for_pdf(pdf_bytes)

    st.header("🔍 Embedding Results")
    for i, item in enumerate(results):
        st.subheader(f"Embedding {i+1}")
        st.markdown(f"Type: {item['type']}")
        st.markdown(f"Vector Length: {len(item['vector'])}")

        # Show partial vector
        st.markdown("Vector (first 10 values):")
        st.code(item['vector'][:10], language='python')

        # Show text content
        if item['type'] == 'text':
            st.markdown("Text Sample:")
            st.text_area("Text", value=item['content'][:1000], height=150)

        # Show image preview
        elif item['type'] == 'image':
            st.markdown(f"From Page: {item['page']}")
            st.image(io.BytesIO(base64.b64decode(item['base64'])), use_container_width=True)
            width=150

        st.markdown("---")
