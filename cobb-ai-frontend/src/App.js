import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, Activity, Brain, AlertCircle, CheckCircle, Loader, Sparkles, X } from 'lucide-react';

const CobbAIChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'ai',
      content: "Hello! I'm your Cobb Angle AI Assistant. I can analyze spine X-ray images to detect scoliosis and measure Cobb angles. Upload an X-ray image to get started!",
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (type, content, data = null) => {
    const newMessage = {
      id: Date.now(),
      type,
      content,
      data,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        addMessage('user', `Uploaded X-ray image: ${file.name}`, {
          type: 'image',
          preview: e.target.result
        });
        
        setTimeout(() => analyzeImage(file), 500);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async (file) => {
    setIsAnalyzing(true);
    setIsTyping(true);

    addMessage('ai', 'Analyzing your X-ray image... This may take a few moments.', {
      type: 'status',
      status: 'analyzing'
    });

    try {
      const formData = new FormData();
      formData.append('files', file);

      const response = await fetch('http://127.0.0.1:8000/predict_cobb', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      const result = data.results[0];

      setIsTyping(false);
      setIsAnalyzing(false);

      addMessage('ai', 'Analysis complete! Here are the results:', {
        type: 'results',
        thoracic: result.thoracic_cobb_deg,
        lumbar: result.lumbar_cobb_deg,
        filename: result.filename
      });

    } catch (error) {
      setIsTyping(false);
      setIsAnalyzing(false);
      addMessage('ai', `Sorry, I encountered an error: ${error.message}. Please make sure the backend is running and try again.`, {
        type: 'error'
      });
    }
  };

  const handleSendMessage = () => {
    if (!inputText.trim()) return;

    addMessage('user', inputText);
    const userMessage = inputText.toLowerCase();
    setInputText('');
    setIsTyping(true);

    setTimeout(() => {
      let response = '';
      
      if (userMessage.includes('hello') || userMessage.includes('hi')) {
        response = "Hello! Ready to analyze spine X-rays. Please upload an image to get started.";
      } else if (userMessage.includes('how') && userMessage.includes('work')) {
        response = "I use advanced deep learning models (U-Net for segmentation and ResNet50 for angle detection) to analyze spine X-rays. Just upload an image, and I'll measure the thoracic and lumbar Cobb angles automatically!";
      } else if (userMessage.includes('cobb') || userMessage.includes('angle')) {
        response = "The Cobb angle is the standard measurement for assessing scoliosis severity. It measures the angle of spinal curvature. Angles >10° indicate scoliosis, with >25° often requiring treatment.";
      } else if (userMessage.includes('upload') || userMessage.includes('image')) {
        response = "Click the upload button (📎 icon) to select your X-ray image. I support PNG, JPG, and JPEG formats.";
      } else {
        response = "I'm specialized in analyzing spine X-rays for Cobb angle measurement. Upload an X-ray image to get started, or ask me about how the analysis works!";
      }
      
      setIsTyping(false);
      addMessage('ai', response);
    }, 1000);
  };

  const getSeverityInfo = (thoracic, lumbar) => {
    const maxAngle = Math.max(thoracic, lumbar);
    
    if (maxAngle < 10) {
      return { level: 'Normal', color: '#059669', bg: '#d1fae5' };
    } else if (maxAngle < 25) {
      return { level: 'Mild Scoliosis', color: '#d97706', bg: '#fef3c7' };
    } else if (maxAngle < 40) {
      return { level: 'Moderate Scoliosis', color: '#ea580c', bg: '#fed7aa' };
    } else {
      return { level: 'Severe Scoliosis', color: '#dc2626', bg: '#fee2e2' };
    }
  };

  const renderMessage = (message) => {
    const messageContainerStyle = {
      display: 'flex',
      justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start',
      marginBottom: '1rem',
      animation: 'fadeIn 0.3s ease-out'
    };

    if (message.type === 'user') {
      return (
        <div key={message.id} style={messageContainerStyle}>
          <div style={{ maxWidth: '70%' }}>
            {message.data?.type === 'image' && (
              <div style={{ marginBottom: '0.5rem' }}>
                <img 
                  src={message.data.preview} 
                  alt="Uploaded X-ray" 
                  style={{
                    borderRadius: '12px',
                    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                    maxHeight: '256px',
                    width: 'auto'
                  }}
                />
              </div>
            )}
            <div style={{
              background: 'linear-gradient(135deg, #2563eb 0%, #1e40af 100%)',
              color: 'white',
              borderRadius: '18px',
              borderTopRightRadius: '4px',
              padding: '12px 16px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}>
              <p style={{ fontSize: '14px', margin: 0 }}>{message.content}</p>
            </div>
            <p style={{ fontSize: '11px', color: '#9ca3af', marginTop: '4px', textAlign: 'right' }}>
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </p>
          </div>
        </div>
      );
    }

    return (
      <div key={message.id} style={messageContainerStyle}>
        <div style={{ display: 'flex', flexShrink: 0, marginRight: '12px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #a855f7 0%, #3b82f6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
          }}>
            <Brain size={20} color="white" />
          </div>
        </div>
        <div style={{ maxWidth: '70%' }}>
          <div style={{
            background: 'white',
            borderRadius: '18px',
            borderTopLeftRadius: '4px',
            padding: '12px 16px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
            border: '1px solid #e5e7eb'
          }}>
            {message.data?.type === 'results' ? (
              <div>
                <p style={{ fontSize: '14px', color: '#374151', fontWeight: '500', marginBottom: '12px' }}>
                  {message.content}
                </p>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {/* Thoracic */}
                  <div style={{
                    background: 'linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%)',
                    borderRadius: '12px',
                    padding: '16px',
                    border: '1px solid #bfdbfe'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ fontSize: '11px', fontWeight: '600', color: '#2563eb', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        Thoracic Cobb Angle
                      </span>
                      <Activity size={16} color="#2563eb" />
                    </div>
                    <span style={{ fontSize: '28px', fontWeight: 'bold', color: '#1e3a8a' }}>
                      {message.data.thoracic.toFixed(1)}°
                    </span>
                  </div>

                  {/* Lumbar */}
                  <div style={{
                    background: 'linear-gradient(135deg, #fae8ff 0%, #fce7f3 100%)',
                    borderRadius: '12px',
                    padding: '16px',
                    border: '1px solid #e9d5ff'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ fontSize: '11px', fontWeight: '600', color: '#9333ea', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        Lumbar Cobb Angle
                      </span>
                      <Activity size={16} color="#9333ea" />
                    </div>
                    <span style={{ fontSize: '28px', fontWeight: 'bold', color: '#581c87' }}>
                      {message.data.lumbar.toFixed(1)}°
                    </span>
                  </div>

                  {/* Severity */}
                  {(() => {
                    const severity = getSeverityInfo(message.data.thoracic, message.data.lumbar);
                    const Icon = severity.level === 'Normal' ? CheckCircle : AlertCircle;
                    return (
                      <div style={{
                        backgroundColor: severity.bg,
                        borderRadius: '12px',
                        padding: '16px',
                        border: '1px solid #e5e7eb'
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <Icon size={20} color={severity.color} />
                          <div>
                            <p style={{ fontSize: '11px', color: '#6b7280', fontWeight: '500', margin: 0 }}>Assessment</p>
                            <p style={{ fontSize: '18px', fontWeight: 'bold', color: severity.color, margin: 0 }}>
                              {severity.level}
                            </p>
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </div>

                <div style={{ paddingTop: '12px', borderTop: '1px solid #e5e7eb', marginTop: '12px' }}>
                  <p style={{ fontSize: '11px', color: '#6b7280', fontStyle: 'italic', margin: 0 }}>
                    Note: This is an AI-generated analysis. Please consult a healthcare professional for medical diagnosis and treatment decisions.
                  </p>
                </div>
              </div>
            ) : message.data?.type === 'status' ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Loader size={20} color="#3b82f6" style={{ animation: 'spin 1s linear infinite' }} />
                <p style={{ fontSize: '14px', color: '#374151', margin: 0 }}>{message.content}</p>
              </div>
            ) : message.data?.type === 'error' ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <AlertCircle size={20} color="#ef4444" />
                <p style={{ fontSize: '14px', color: '#374151', margin: 0 }}>{message.content}</p>
              </div>
            ) : (
              <p style={{ fontSize: '14px', color: '#374151', margin: 0 }}>{message.content}</p>
            )}
          </div>
          <p style={{ fontSize: '11px', color: '#9ca3af', marginTop: '4px' }}>
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </p>
        </div>
      </div>
    );
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #f8fafc 0%, #dbeafe 50%, #e0e7ff 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '16px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    }}>
      <div style={{
        width: '100%',
        maxWidth: '1000px',
        height: '90vh',
        background: 'white',
        borderRadius: '24px',
        boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        border: '1px solid #e5e7eb'
      }}>
        
        {/* Header */}
        <div style={{
          background: 'linear-gradient(135deg, #2563eb 0%, #4f46e5 50%, #7c3aed 100%)',
          color: 'white',
          padding: '20px 24px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '40px',
                height: '40px',
                borderRadius: '50%',
                background: 'rgba(255,255,255,0.2)',
                backdropFilter: 'blur(10px)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <Brain size={24} />
              </div>
              <div>
                <h1 style={{ fontSize: '20px', fontWeight: 'bold', margin: 0 }}>Cobb Angle AI</h1>
                <p style={{ fontSize: '12px', color: '#bfdbfe', margin: 0 }}>Intelligent Scoliosis Detection</p>
              </div>
            </div>
            <Sparkles size={24} color="#fde047" style={{ animation: 'pulse 2s infinite' }} />
          </div>
        </div>

        {/* Messages Area */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '24px',
          background: 'linear-gradient(to bottom, rgba(249,250,251,0.5) 0%, transparent 100%)'
        }}>
          {messages.map(renderMessage)}
          
          {isTyping && (
            <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', flexShrink: 0, marginRight: '12px' }}>
                <div style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #a855f7 0%, #3b82f6 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                }}>
                  <Brain size={20} color="white" />
                </div>
              </div>
              <div style={{
                background: 'white',
                borderRadius: '18px',
                borderTopLeftRadius: '4px',
                padding: '12px 16px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
                border: '1px solid #e5e7eb'
              }}>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <div style={{ width: '8px', height: '8px', background: '#9ca3af', borderRadius: '50%', animation: 'bounce 1s infinite' }}></div>
                  <div style={{ width: '8px', height: '8px', background: '#9ca3af', borderRadius: '50%', animation: 'bounce 1s infinite 0.15s' }}></div>
                  <div style={{ width: '8px', height: '8px', background: '#9ca3af', borderRadius: '50%', animation: 'bounce 1s infinite 0.3s' }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div style={{
          borderTop: '1px solid #e5e7eb',
          background: 'white',
          padding: '16px'
        }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '12px' }}>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              style={{ display: 'none' }}
            />
            
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalyzing}
              style={{
                flexShrink: 0,
                width: '48px',
                height: '48px',
                borderRadius: '12px',
                background: 'linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)',
                border: 'none',
                cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                transition: 'all 0.2s',
                opacity: isAnalyzing ? 0.5 : 1
              }}
              onMouseEnter={(e) => {
                if (!isAnalyzing) {
                  e.currentTarget.style.background = 'linear-gradient(135deg, #e5e7eb 0%, #d1d5db 100%)';
                  e.currentTarget.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)';
                e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
              }}
            >
              <Upload size={20} color="#374151" />
            </button>

            <div style={{ flex: 1, position: 'relative' }}>
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Ask me anything about Cobb angles or upload an X-ray..."
                disabled={isAnalyzing}
                style={{
                  width: '100%',
                  padding: '12px 40px 12px 16px',
                  borderRadius: '12px',
                  border: '2px solid #e5e7eb',
                  fontSize: '14px',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                  backgroundColor: isAnalyzing ? '#f9fafb' : 'white',
                  cursor: isAnalyzing ? 'not-allowed' : 'text'
                }}
                onFocus={(e) => e.target.style.borderColor = '#3b82f6'}
                onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
              />
              {inputText && (
                <button
                  onClick={() => setInputText('')}
                  style={{
                    position: 'absolute',
                    right: '12px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: '#9ca3af',
                    padding: '4px'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.color = '#6b7280'}
                  onMouseLeave={(e) => e.currentTarget.style.color = '#9ca3af'}
                >
                  <X size={16} />
                </button>
              )}
            </div>

            <button
              onClick={handleSendMessage}
              disabled={!inputText.trim() || isAnalyzing}
              style={{
                flexShrink: 0,
                width: '48px',
                height: '48px',
                borderRadius: '12px',
                background: (!inputText.trim() || isAnalyzing) 
                  ? '#e5e7eb' 
                  : 'linear-gradient(135deg, #2563eb 0%, #4f46e5 100%)',
                border: 'none',
                cursor: (!inputText.trim() || isAnalyzing) ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                transition: 'all 0.2s',
                opacity: (!inputText.trim() || isAnalyzing) ? 0.5 : 1
              }}
              onMouseEnter={(e) => {
                if (inputText.trim() && !isAnalyzing) {
                  e.currentTarget.style.background = 'linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%)';
                  e.currentTarget.style.boxShadow = '0 4px 6px rgba(0,0,0,0.15)';
                }
              }}
              onMouseLeave={(e) => {
                if (inputText.trim() && !isAnalyzing) {
                  e.currentTarget.style.background = 'linear-gradient(135deg, #2563eb 0%, #4f46e5 100%)';
                  e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                }
              }}
            >
              <Send size={20} color={(!inputText.trim() || isAnalyzing) ? '#9ca3af' : 'white'} />
            </button>
          </div>

          <p style={{ fontSize: '11px', color: '#9ca3af', marginTop: '12px', textAlign: 'center' }}>
            Upload spine X-ray images for AI-powered Cobb angle analysis
          </p>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-8px); }
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
};

export default CobbAIChatInterface;