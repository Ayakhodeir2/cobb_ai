import dynamic from 'next/dynamic';

const ClinicalCobbInterface = dynamic(
  () => import('../components/ClinicalCobbInterface'),
  { 
    ssr: false,
    loading: () => (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        fontSize: '18px' 
      }}>
        Loading Clinical Interface...
      </div>
    )
  }
);

export default function Home() {
  return <ClinicalCobbInterface />;
}