# 🎵 Perch 2.0 Integration Setup Guide

## Overview
This integration brings Google DeepMind's **Perch 2.0 Bird Vocalization Classifier** to your bird hotspot analysis application.

---

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install librosa soundfile plotly streamlit-webrtc av kaggle
```

### 2. Kaggle API Setup (for Perch 2.0 Model)
1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account → API → Create New Token
3. Download `kaggle.json` and place it in:
   - **Windows**: `C:\Users\{username}\.kaggle\kaggle.json`
   - **macOS/Linux**: `~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### 3. Run the Application
```bash
streamlit run bird_hotspot_ui.py
```

Navigate to **"🎵 Perch 2.0 Bird Sound Testing"** in the sidebar.

---

## 🎯 Features

### 📁 Audio File Upload
- Support for MP3, WAV, M4A, FLAC, OGG formats
- Real-time spectrogram visualization
- Perch 2.0 AI classification with confidence scores
- Scientific name identification

### 🎙️ Live Recording
- Real-time microphone input
- 3-second rolling spectrogram
- Auto-classification as you record
- Perfect for testing with:
  - Bird sounds from your garden
  - YouTube bird song videos
  - Field recordings

---

## 🔧 Technical Details

### Model Integration
- **Model Source**: `google/bird-vocalization-classifier` on Kaggle
- **Download Size**: ~100MB
- **Supported Species**: 1000+ bird species globally
- **Input Format**: 22kHz mono audio
- **Output**: Species predictions with confidence scores

### Real-time Processing
- **Spectrogram**: 3-second sliding window
- **Classification**: Every 2+ seconds of audio
- **Latency**: <1 second for inference
- **Memory**: Efficient streaming processing

---

## 🎵 Usage Examples

### Test with YouTube
1. Open YouTube bird song video
2. Enable "🎙️ Live Recording" 
3. Play the video near your microphone
4. Watch real-time classification results

### Garden Recording
1. Take your laptop to the garden
2. Enable real-time recording
3. See live bird identifications
4. Export results for your bird log

### File Analysis
1. Upload existing bird recordings
2. View detailed spectrogram analysis
3. Get scientific species identification
4. Compare with eBird hotspot data

---

## 🔮 Future Enhancements

### Phase 2: Advanced Features
- [ ] **Model Ensemble**: Combine Perch 2.0 with BirdNET
- [ ] **Regional Filtering**: India-specific bird species focus
- [ ] **Confidence Tuning**: Adjust sensitivity for local conditions
- [ ] **Batch Processing**: Analyze multiple files simultaneously

### Phase 3: Field Integration
- [ ] **Mobile App**: Streamlit mobile-optimized interface
- [ ] **GPS Integration**: Auto-tag recordings with location
- [ ] **eBird Integration**: Submit findings directly to eBird
- [ ] **Hotspot Validation**: Cross-check with known hotspot data

---

## 🐛 Troubleshooting

### Common Issues

**❌ "Audio libraries not installed"**
```bash
pip install librosa soundfile plotly
```

**❌ "Kaggle API not available"**
```bash
pip install kaggle
# Then setup kaggle.json as described above
```

**❌ "Real-time recording not working"**
```bash
pip install streamlit-webrtc av
# Allow microphone access in browser
```

**❌ "Model download failed"**
- Check internet connection
- Verify Kaggle API credentials
- Ensure sufficient disk space (200MB)

---

## 📞 Support

For issues specific to:
- **Perch 2.0 Model**: Check the [Kaggle model page](https://www.kaggle.com/models/google/bird-vocalization-classifier)
- **Audio Processing**: Refer to [librosa documentation](https://librosa.org/)
- **Real-time Streaming**: Check [streamlit-webrtc docs](https://github.com/whitphx/streamlit-webrtc)

---

**🎵 Happy Bird Listening with Perch 2.0! 🐦**

