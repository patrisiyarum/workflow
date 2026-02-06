import React, { useCallback, useRef, useState } from 'react';
import { Upload, Image as ImageIcon, Film } from 'lucide-react';
import './UploadZone.css';

export default function UploadZone({ onUpload }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFile = useCallback(
    (file) => {
      if (file) onUpload(file);
    },
    [onUpload]
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer?.files?.[0];
      handleFile(file);
    },
    [handleFile]
  );

  const onDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const onDragLeave = () => setDragActive(false);

  return (
    <div
      className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*,video/*"
        hidden
        onChange={(e) => handleFile(e.target.files?.[0])}
      />

      <div className="upload-icon-group">
        <ImageIcon size={28} className="upload-icon-img" />
        <Upload size={36} className="upload-icon-main" />
        <Film size={28} className="upload-icon-vid" />
      </div>

      <h3>Drop an OR image or video here</h3>
      <p>or click to browse — JPG, PNG, MP4, AVI supported</p>

      <div className="upload-badges">
        <span className="badge">Single Image</span>
        <span className="badge">Video File</span>
      </div>
    </div>
  );
}
