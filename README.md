# Midi RAG Web Interface

A Streamlit web application providing both patient and clinical RAG interfaces for Midi Health.

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd midi_rag_web
   pip install -r requirements.txt
   ```

2. **Configure Secrets**
   ```bash
   cp .streamlit/secrets.toml.template .streamlit/secrets.toml
   # Edit .streamlit/secrets.toml with your actual API keys and passwords
   ```

3. **Run Locally**
   ```bash
   streamlit run streamlit_app.py
   ```

### Deploy to Streamlit Community Cloud

1. **Push to GitHub**
   - Create a new repository on GitHub
   - Push this code to your repository

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Choose `streamlit_app.py` as the main file
   - Add your secrets in the deployment settings

3. **Configure Secrets in Streamlit Cloud**
   ```toml
   ANTHROPIC_API_KEY = "your-actual-api-key"
   PATIENT_PASSWORD = "your-patient-password"
   CLINICAL_PASSWORD = "your-clinical-password"
   ```

## 🔐 Authentication

The app includes password protection for both interfaces:

- **Patient Interface**: For patient education and support
- **Clinical Interface**: For healthcare providers accessing clinical protocols

Default passwords (change these in production):
- Patient: `midi-patient-2025`
- Clinical: `midi-clinical-2025`

## 📁 Project Structure

```
midi_rag_web/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── src/                      # RAG engine source code
│   ├── rag_engine.py        # Clinical RAG engine
│   └── patient_rag_engine.py # Patient RAG engine
├── data/                     # Vector store data
│   └── chroma_db/           # ChromaDB vector database
└── .streamlit/              # Streamlit configuration
    ├── config.toml          # App configuration
    └── secrets.toml.template # Secrets template
```

## 🎯 Features

### Patient Interface
- 💬 Conversational health support
- 📚 Educational information about Midi services
- 🔒 Password protected access
- 💾 Chat history within session

### Clinical Interface
- 🏥 Clinical protocol searches
- 📊 Confidence scoring for answers
- 📋 Source documentation
- 🔍 Evidence-based guidelines
- 💾 Query history within session

## 🛠️ Technical Details

- **Framework**: Streamlit
- **LLM**: Anthropic Claude Sonnet 4
- **Vector Store**: ChromaDB
- **Documents**: 4,726+ protocol documents
- **Embeddings**: Sentence Transformers

## 🔧 Environment Variables

Required in `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your-anthropic-api-key"
PATIENT_PASSWORD = "your-patient-password"
CLINICAL_PASSWORD = "your-clinical-password"
```

## 📝 Usage

1. **Access the App**: Open your deployed URL or run locally
2. **Choose Interface**: Select Patient or Clinical from the sidebar
3. **Authenticate**: Enter the appropriate password
4. **Ask Questions**: Use the text area to submit questions
5. **View Responses**: See answers with confidence scores (clinical) or conversational responses (patient)

## 🚀 Deployment Options

### Streamlit Community Cloud (Recommended)
- Free hosting for public repositories
- Automatic deployments from GitHub
- Built-in secrets management
- Custom subdomain

### Alternative Platforms
- **Hugging Face Spaces**: Free ML app hosting
- **Railway**: Simple deployment with PostgreSQL
- **Render**: Web service hosting
- **Heroku**: Platform-as-a-service

## 🔍 Troubleshooting

### Common Issues

1. **ChromaDB Loading Error**
   - Ensure `data/chroma_db` directory exists
   - Check file permissions

2. **API Key Issues**
   - Verify `ANTHROPIC_API_KEY` in secrets
   - Test API key with a simple request

3. **Password Protection**
   - Check `secrets.toml` configuration
   - Clear browser cache/session storage

### Debug Mode

Run with debug logging:
```bash
streamlit run streamlit_app.py --logger.level=debug
```

## 📞 Support

For issues with this web interface, check:
1. Streamlit logs in the deployment console
2. Browser developer console for client-side errors
3. GitHub repository issues

## 🔄 Updates

To update the deployment:
1. Push changes to your GitHub repository
2. Streamlit Community Cloud will automatically redeploy
3. Monitor the deployment logs for any issues

---

Built with ❤️ for Midi Health team