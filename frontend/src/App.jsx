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
      setError('Unsupported file type. Use JPG, PNG, MP4, or AVI.');
      return;
    }

    if (isImage) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
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
      setError(err.message || 'Prediction failed.');
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
                  Upload an operating room image or video to identify the
                  surgical phase using a ResNet + LSTM model.
                </p>
              </section>

              <UploadZone onUpload={handleUpload} />
              <PhaseInfo />
            </>
          )}

          {loading && (
            <div className="loading-state">
              <div className="spinner" />
              <p>Analyzing...</p>
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
