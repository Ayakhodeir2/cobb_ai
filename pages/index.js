import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';

const ClinicalCobbInterface = dynamic(
  () => import('./components/ClinicalCobbInterface'),
  { 
    ssr: false,
    loading: () => (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        fontSize: '18px',
        fontFamily: 'system-ui, sans-serif',
        color: '#64748b'
      }}>
        Loading Clinical Interface...
      </div>
    )
  }
);

export default function Home() {
  const [error, setError] = useState(null);

  useEffect(() => {
    const handleError = (e) => {
      console.error('Error:', e);
      setError(e.message);
    };
    
    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, []);

  if (error) {
    return (
      <div style={{
        padding: '40px',
        fontFamily: 'system-ui',
        maxWidth: '800px',
        margin: '0 auto'
      }}>
        <h1 style={{ color: '#ef4444' }}>Error Loading Component</h1>
        <pre style={{ 
          background: '#fee2e2', 
          padding: '20px', 
          borderRadius: '8px',
          overflow: 'auto'
        }}>
          {error}
        </pre>
      </div>
    );
  }

  return <ClinicalCobbInterface />;
}