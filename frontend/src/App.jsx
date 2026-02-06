import React, { useState, useCallback } from 'react';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import ImageResult from './components/ImageResult';
import VideoResult from './components/VideoResult';
import PhaseInfo from './components/PhaseInfo';
import Footer from './components/Footer';
import { predictImage, predictVideo } from './api';
import './App.css';

export default function App() {
  const [mode, setMode] = useState('image'); // 'image' | 'video'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [videoResult, setVideoResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleUpload = useCallback(async (file) => {
    setError(null);
    setImageResult(null);
    setVideoResult(null);

    const isVideo = file.type.startsWith('video/');
    const isImage = file.type.startsWith('image/');

    if (!isVideo && !isImage) {
      setError('Please upload an image (JPG, PNG) or video (MP4, AVI) file.');
      return;
    }

    if (isImage) {
      setMode('image');
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setMode('video');
      setPreviewUrl(null);
    }

    setLoading(true);
    try {
      if (isImage) {
        const result = await predictImage(file);
        setImageResult(result);
      } else {
        const result = await predictVideo(file, 25, 200);
        setVideoResult(result);
      }
    } catch (err) {
      setError(err.message || 'Prediction failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleReset = () => {
    setImageResult(null);
    setVideoResult(null);
    setPreviewUrl(null);
    setError(null);
  };

  const hasResult = imageResult || videoResult;

  return (
    <div className="app">
      <Header />

      <main className="main">
        <div className="container">
          {!hasResult && !loading && (
            <>
              <section className="hero">
                <h1>Surgery Phase Detection</h1>
                <p>
                  Upload an operating room image or video to detect the current
                  surgical phase using deep learning. Powered by a ResNet + LSTM
                  architecture trained on the MVOR dataset.
                </p>
              </section>

              <UploadZone onUpload={handleUpload} />
              <PhaseInfo />
            </>
          )}

          {loading && (
            <div className="loading-state">
              <div className="spinner" />
              <h2>Analyzing...</h2>
              <p>Running inference through the neural network</p>
            </div>
          )}

          {error && (
            <div className="error-banner">
              <span>{error}</span>
              <button onClick={() => setError(null)}>Dismiss</button>
            </div>
          )}

          {imageResult && !loading && (
            <ImageResult
              result={imageResult}
              previewUrl={previewUrl}
              onReset={handleReset}
            />
          )}

          {videoResult && !loading && (
            <VideoResult result={videoResult} onReset={handleReset} />
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}
