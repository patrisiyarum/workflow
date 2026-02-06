import React, { useCallback, useRef, useState } from 'react';
import { Upload } from 'lucide-react';
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

  return (
    <div
      className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
      onDrop={onDrop}
      onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
      onDragLeave={() => setDragActive(false)}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*,video/*"
        hidden
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      <Upload size={20} className="upload-icon" />
      <span className="upload-text">
        Drop a file here or <span className="upload-link">browse</span>
      </span>
      <span className="upload-hint">JPG, PNG, MP4, AVI</span>
    </div>
  );
}
