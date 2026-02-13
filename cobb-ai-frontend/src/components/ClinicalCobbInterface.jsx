import React, { useState, useRef } from 'react';
import { Upload, X, Activity, AlertCircle, CheckCircle, Loader, FileImage } from 'lucide-react';

const ClinicalCobbInterface = () => {
  const [uploadedImages, setUploadedImages] = useState([]);
  const [results, setResults] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef(null);

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    const newImages = files.map(file => ({
      id: Date.now() + Math.random(),
      file,
      preview: URL.createObjectURL(file),
      name: file.name,
      status: 'pending'
    }));
    
    setUploadedImages(prev => [...prev, ...newImages]);
    analyzeImages([...uploadedImages, ...newImages]);
  };

  const removeImage = (id) => {
    setUploadedImages(prev => prev.filter(img => img.id !== id));
    setResults(prev => prev.filter(r => r.imageId !== id));
  };

  const analyzeImages = async (images) => {
    setIsAnalyzing(true);
    
    const pendingImages = images.filter(img => img.status === 'pending');
    
    for (const image of pendingImages) {
      try {
        const formData = new FormData();
        formData.append('files', image.file);

        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
        const response = await fetch(`${apiUrl}/predict_cobb`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error('Analysis failed');
        }

        const data = await response.json();
        const result = data.results[0];

        const thoracic = Math.max(0, result.thoracic_cobb_deg || result.thoracic || 0);
        const lumbar = Math.max(0, result.lumbar_cobb_deg || result.lumbar || 0);

        setResults(prev => [...prev, {
          imageId: image.id,
          filename: result.filename,
          thoracic: thoracic,
          lumbar: lumbar,
          timestamp: new Date()
        }]);

        setUploadedImages(prev => prev.map(img => 
          img.id === image.id ? { ...img, status: 'completed' } : img
        ));

      } catch (error) {
        console.error('Analysis error:', error);
        setUploadedImages(prev => prev.map(img => 
          img.id === image.id ? { ...img, status: 'error' } : img
        ));
      }
    }
    
    setIsAnalyzing(false);
  };

  const getSeverityLevel = (thoracic, lumbar) => {
    const maxAngle = Math.max(thoracic, lumbar);
    
    if (maxAngle < 10) {
      return { level: 'Normal', color: '#10b981', bg: '#d1fae5', textColor: '#065f46' };
    } else if (maxAngle < 25) {
      return { level: 'Mild Scoliosis', color: '#f59e0b', bg: '#fef3c7', textColor: '#92400e' };
    } else if (maxAngle < 40) {
      return { level: 'Moderate Scoliosis', color: '#f97316', bg: '#ffedd5', textColor: '#9a3412' };
    } else {
      return { level: 'Severe Scoliosis', color: '#ef4444', bg: '#fee2e2', textColor: '#991b1b' };
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: '#f8fafc',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      padding: '24px'
    }}>
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto'
      }}>
        
        {/* Header */}
        <div style={{
          background: 'white',
          borderRadius: '12px',
          padding: '24px',
          marginBottom: '24px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          borderLeft: '4px solid #3b82f6'
        }}>
          <div>
            <h1 style={{ 
              fontSize: '28px', 
              fontWeight: '700', 
              color: '#1e293b', 
              margin: '0 0 8px 0' 
            }}>
              Cobb Angle Measurement System
            </h1>
            <p style={{ 
              fontSize: '14px', 
              color: '#64748b', 
              margin: 0 
            }}>
              AI-powered scoliosis detection and measurement tool for clinical use
            </p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '24px' }}>
          
          {/* Main Content */}
          <div>
            {/* Upload Section */}
            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '32px',
              marginBottom: '24px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              textAlign: 'center'
            }}>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleImageUpload}
                style={{ display: 'none' }}
              />
              
              <div style={{
                border: '2px dashed #cbd5e1',
                borderRadius: '12px',
                padding: '48px 24px',
                cursor: 'pointer',
                transition: 'all 0.2s',
                background: '#f8fafc'
              }}
              onClick={() => fileInputRef.current?.click()}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = '#3b82f6';
                e.currentTarget.style.background = '#eff6ff';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#cbd5e1';
                e.currentTarget.style.background = '#f8fafc';
              }}>
                <Upload size={48} color="#3b82f6" style={{ margin: '0 auto 16px' }} />
                <h3 style={{ fontSize: '18px', fontWeight: '600', color: '#1e293b', margin: '0 0 8px 0' }}>
                  Upload X-ray Images
                </h3>
                <p style={{ fontSize: '14px', color: '#64748b', margin: 0 }}>
                  Click to browse or drag and drop • Multiple images supported
                </p>
              </div>
            </div>

            {/* Results Grid */}
            {uploadedImages.length > 0 && (
              <div style={{
                background: 'white',
                borderRadius: '12px',
                padding: '24px',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
              }}>
                <h2 style={{ fontSize: '20px', fontWeight: '600', color: '#1e293b', margin: '0 0 20px 0' }}>
                  Analysis Results
                </h2>
                
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '20px' }}>
                  {uploadedImages.map(image => {
                    const result = results.find(r => r.imageId === image.id);
                    const severity = result ? getSeverityLevel(result.thoracic, result.lumbar) : null;
                    
                    return (
                      <div key={image.id} style={{
                        border: '1px solid #e2e8f0',
                        borderRadius: '12px',
                        overflow: 'hidden',
                        transition: 'all 0.2s'
                      }}>
                        {/* Image Preview */}
                        <div style={{ position: 'relative', background: '#000' }}>
                          <img 
                            src={image.preview} 
                            alt={image.name}
                            style={{
                              width: '100%',
                              height: '200px',
                              objectFit: 'contain'
                            }}
                          />
                          <button
                            onClick={() => removeImage(image.id)}
                            style={{
                              position: 'absolute',
                              top: '8px',
                              right: '8px',
                              width: '32px',
                              height: '32px',
                              borderRadius: '50%',
                              background: 'rgba(0,0,0,0.6)',
                              border: 'none',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              transition: 'all 0.2s'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(239,68,68,0.9)'}
                            onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(0,0,0,0.6)'}
                          >
                            <X size={18} color="white" />
                          </button>
                        </div>

                        {/* Results */}
                        <div style={{ padding: '16px' }}>
                          <div style={{ 
                            fontSize: '12px', 
                            color: '#64748b', 
                            marginBottom: '12px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px'
                          }}>
                            <FileImage size={14} />
                            {image.name}
                          </div>

                          {image.status === 'pending' && (
                            <div style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '12px',
                              padding: '16px',
                              background: '#eff6ff',
                              borderRadius: '8px'
                            }}>
                              <Loader size={20} color="#3b82f6" style={{ animation: 'spin 1s linear infinite' }} />
                              <span style={{ fontSize: '14px', color: '#1e40af' }}>Analyzing...</span>
                            </div>
                          )}

                          {image.status === 'error' && (
                            <div style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '12px',
                              padding: '16px',
                              background: '#fee2e2',
                              borderRadius: '8px'
                            }}>
                              <AlertCircle size={20} color="#ef4444" />
                              <span style={{ fontSize: '14px', color: '#991b1b' }}>Analysis failed</span>
                            </div>
                          )}

                          {result && (
                            <div>
                              {/* Measurements */}
                              <div style={{ display: 'flex', gap: '12px', marginBottom: '12px' }}>
                                <div style={{
                                  flex: 1,
                                  background: '#eff6ff',
                                  padding: '12px',
                                  borderRadius: '8px',
                                  border: '1px solid #bfdbfe'
                                }}>
                                  <div style={{ 
                                    fontSize: '11px', 
                                    color: '#1e40af', 
                                    fontWeight: '600',
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.5px',
                                    marginBottom: '4px'
                                  }}>
                                    Thoracic
                                  </div>
                                  <div style={{ 
                                    fontSize: '24px', 
                                    fontWeight: '700', 
                                    color: '#1e3a8a',
                                    display: 'flex',
                                    alignItems: 'baseline',
                                    gap: '4px'
                                  }}>
                                    {result.thoracic.toFixed(1)}
                                    <span style={{ fontSize: '16px', fontWeight: '500' }}>°</span>
                                  </div>
                                </div>

                                <div style={{
                                  flex: 1,
                                  background: '#faf5ff',
                                  padding: '12px',
                                  borderRadius: '8px',
                                  border: '1px solid #e9d5ff'
                                }}>
                                  <div style={{ 
                                    fontSize: '11px', 
                                    color: '#7c3aed', 
                                    fontWeight: '600',
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.5px',
                                    marginBottom: '4px'
                                  }}>
                                    Lumbar
                                  </div>
                                  <div style={{ 
                                    fontSize: '24px', 
                                    fontWeight: '700', 
                                    color: '#581c87',
                                    display: 'flex',
                                    alignItems: 'baseline',
                                    gap: '4px'
                                  }}>
                                    {result.lumbar.toFixed(1)}
                                    <span style={{ fontSize: '16px', fontWeight: '500' }}>°</span>
                                  </div>
                                </div>
                              </div>

                              {/* Severity Badge */}
                              <div style={{
                                background: severity.bg,
                                padding: '12px',
                                borderRadius: '8px',
                                border: `1px solid ${severity.color}`,
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px'
                              }}>
                                {severity.level === 'Normal' ? (
                                  <CheckCircle size={18} color={severity.color} />
                                ) : (
                                  <AlertCircle size={18} color={severity.color} />
                                )}
                                <div style={{ flex: 1 }}>
                                  <div style={{ fontSize: '11px', color: severity.textColor, fontWeight: '500' }}>
                                    Assessment
                                  </div>
                                  <div style={{ fontSize: '14px', fontWeight: '700', color: severity.textColor }}>
                                    {severity.level}
                                  </div>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Info Notice */}
            {uploadedImages.length === 0 && (
              <div style={{
                background: 'white',
                borderRadius: '12px',
                padding: '24px',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
              }}>
                <h3 style={{ fontSize: '16px', fontWeight: '600', color: '#1e293b', margin: '0 0 12px 0' }}>
                  About This Tool
                </h3>
                <ul style={{ fontSize: '14px', color: '#64748b', lineHeight: '1.8', margin: 0, paddingLeft: '20px' }}>
                  <li>Automated Cobb angle measurement using deep learning (U-Net + ResNet50)</li>
                  <li>Measures both thoracic and lumbar curvature from AP spine X-rays</li>
                  <li>Results typically available within 2-3 seconds per image</li>
                  <li>All measurements should be verified by a qualified healthcare professional</li>
                  <li>Supports batch processing of multiple images for efficient workflow</li>
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* Footer Disclaimer */}
        <div style={{
          background: '#fff7ed',
          border: '1px solid #fed7aa',
          borderRadius: '12px',
          padding: '16px',
          marginTop: '24px',
          display: 'flex',
          gap: '12px'
        }}>
          <AlertCircle size={20} color="#ea580c" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <p style={{ fontSize: '13px', fontWeight: '600', color: '#9a3412', margin: '0 0 4px 0' }}>
              Medical Disclaimer
            </p>
            <p style={{ fontSize: '13px', color: '#9a3412', margin: 0, lineHeight: '1.6' }}>
              This AI tool is intended for screening and measurement assistance only. All measurements must be verified by a qualified healthcare professional. Do not use as the sole basis for clinical decisions.
            </p>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default ClinicalCobbInterface;
