# AI Multimodal Smart Knowledge Assistant — Live Demo Script

Use this script during live presentations and grading to demonstrate all core modalities and features.

---

## Preparation
1. Ensure your `.env` file is configured with active keys (`GROQ_API_KEY`, `GEMINI_API_KEY`).
2. Run the application:
   ```bash
   python ui/gradio_app.py
   ```
3. Open your browser to the local server URL: `http://127.0.0.1:7860`.

---

## Scenario 1: Ask (Text & Voice Q&A)

### Step 1: Text Query Verification
1. Navigate to the **💬 Ask (Text & Voice)** tab.
2. Select **College** from the *Search Domain Filters* dropdown.
3. In the *Typed Question* text box, enter:
   `What is the course code and credits for Introduction to Computer Science?`
4. Click **Submit Question**.
5. **Expected Output:**
   - **Answer Text:** *"The course code for Introduction to Computer Science is CS-101, and it is worth 4 credits."*
   - **LLM Provider:** `groq` (or `bob` if active, or `gemini` if fallback occurred)
   - **Documents Used:** `course_catalog.txt` (or comma-separated list including it)
   - **Voice Answer Spoken:** Plays the synthesized reading of the answer.

### Step 2: Voice Query Verification
1. Click the microphone icon in the *Voice Recording* card and record yourself saying:
   `What are the benefits of crop rotation?`
   *(Alternatively, upload a sample audio query).*
2. Set the *Search Domain Filters* to **Auto-detect**.
3. Click **Submit Question**.
4. **Expected Output:**
   - **Typed Question:** Automatically populates with: *"What are the benefits of crop rotation?"* (via STT)
   - **Answer Text:** *"The benefits of crop rotation include: breaking pest/disease cycles..."*
   - **Documents Used:** `crop_rotation.txt, irrigation_guide.txt`
   - **LLM Provider:** `groq`
   - **Voice Answer Spoken:** Plays back the answer audio synthesized via gTTS.

---

## Scenario 2: Image Captioning & Follow-Up

1. Navigate to the **🖼️ Image Captioning** tab.
2. Drag and drop any landscape or object image (e.g. one of the monuments or farm images).
3. Set the *Description Style* to **detailed**.
4. Click **Generate Caption**.
5. **Expected Output:**
   - **Caption Result:** Shows a paragraph describing the uploaded image (e.g., *"A close-up shot of agricultural crops growing in a field under clear blue skies..."*).
   - **Vision Provider:** `groq` (or `gemini`).
   - **Latency:** Displays the API call response time.
6. Click **Learn More about this Caption via RAG**.
7. **Expected Output:**
   - **RAG Context Analysis:** Displays detailed information retrieved from our knowledge base relating to the caption elements (e.g. details about crop irrigation, soil maintenance).
   - **RAG Sources:** Displays the source files utilized.

---

## Scenario 3: Image Generation

1. Navigate to the **🎨 Image Generation** tab.
2. In the *Image Prompt* box, enter:
   `A gorgeous watercolor painting of a red rose on a rustic wooden desk`
3. Click **Generate Image**.
4. **Expected Output:**
   - **Generated Image:** Displays the rendered rose image.
   - **Gen Provider:** `gemini` (or `pollinations` fallback if the Gemini token quota was exceeded).
