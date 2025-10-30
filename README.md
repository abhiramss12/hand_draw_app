# ✏️ Hand Draw App  

An interactive **hand-drawing web application** built using **Python**, **Streamlit**, and **Virtual Environment**.  
This project lets users draw freely on a digital canvas, save sketches, and (optionally) use AI models for handwriting or sketch recognition.  

---

## 🚀 Features
- 🎨 Interactive drawing canvas  
- 💾 Save your drawings as image files (`.png`)  
- 🧠 (Optional) Connect with AI for sketch or handwriting recognition  
- 🌐 Web interface built with **Streamlit**  
- ⚙️ Runs locally in a **Python virtual environment** for clean dependency management  

---

## 🛠️ Tech Stack
| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Canvas | Streamlit Drawable Canvas |
| Backend | Python |
| Environment | venv (Virtual Environment) |
| Optional ML | TensorFlow / OpenCV / PyTorch |

2. Create a virtual environment
python -m venv venv

3. Activate the environment
🪟 Windows:
venv\Scripts\activate

🐧 Linux / 🍎 macOS:
source venv/bin/activate

4. Install dependencies
pip install -r requirements.txt


Or manually install:

pip install streamlit streamlit-drawable-canvas pillow

5. Run the Streamlit app
streamlit run app.py

6. Open in browser
http://localhost:8501
