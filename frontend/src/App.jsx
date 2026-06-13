import React, { useState, useEffect, useRef } from 'react';
import {
  UploadCloud,
  FileText,
  Database,
  Trash2,
  Send,
  Bot,
  User,
  ShieldCheck,
  Loader2,
  Clock,
  CheckCircle2,
  Plus,
  History,
  BarChart3,
  Settings,
  HelpCircle,
  ChevronDown,
  FlaskConical,
  LogOut,
  ArrowRight
} from 'lucide-react';
import './App.css';
import networkImage from './assets/network_topology.png';

const API_BASE = import.meta.env.VITE_API_BASE || 
  ((window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') 
    ? 'http://127.0.0.1:8000' 
    : 'https://dataset-rag-application.onrender.com');

function App() {
  const [documents, setDocuments] = useState([]);
  const [status, setStatus] = useState({ state: 'checking', text: 'Initializing...' });
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [isIngestionOpen, setIsIngestionOpen] = useState(true);
  const navigateTo = (path) => {
    window.history.pushState(null, null, `/${path}`);
    setView(path);
  };

  const getInitialView = () => {
    const path = window.location.pathname.replace('/', '');
    const validViews = ['knowledge-base', 'query-lab', 'chat-history', 'analytics', 'settings', 'session', 'welcome'];
    return validViews.includes(path) ? path : 'query-lab';
  };

  const [view, setView] = useState(getInitialView());
  const [lastMetrics, setLastMetrics] = useState({ latency: 0, confidence: 0 });
  const [systemInfo, setSystemInfo] = useState({ size_kb: 0, initialized: false });
  const messagesEndRef = useRef(null);

  // Dynamic Settings States
  const [modelName, setModelName] = useState('openai/gpt-4o');
  const [temperature, setTemperature] = useState(0.0);
  const [hybridAlpha, setHybridAlpha] = useState(0.5);
  const [useReranking, setUseReranking] = useState(true);
  const [isSavingSettings, setIsSavingSettings] = useState(false);

  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname.replace('/', '');
      const validViews = ['knowledge-base', 'query-lab', 'chat-history', 'analytics', 'settings', 'session', 'welcome'];
      if (validViews.includes(path)) {
        setView(path);
      } else if (!path || path === '') {
        setView('query-lab');
      }
    };
    window.addEventListener('popstate', handlePopState);
    // Sync initial state if it was empty
    if (window.location.pathname === '/' || window.location.pathname === '') {
      window.history.replaceState(null, null, '/query-lab');
    }
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  useEffect(() => {
    fetchHealthAndDocs();
    fetchSettings();
    const interval = setInterval(fetchHealthAndDocs, 20000); // Poll every 20s
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (messages.length > 0) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const fetchSettings = async () => {
    try {
      const response = await fetch(`${API_BASE}/settings`);
      const data = await response.json();
      setModelName(data.model_name || 'openai/gpt-4o');
      setTemperature(data.temperature !== undefined ? data.temperature : 0.0);
      setHybridAlpha(data.hybrid_alpha !== undefined ? data.hybrid_alpha : 0.5);
      setUseReranking(data.use_reranking !== undefined ? data.use_reranking : true);
    } catch (error) {
      console.error('Failed to fetch settings:', error);
    }
  };

  const handleSaveSettings = async (updates = {}) => {
    setIsSavingSettings(true);
    const nextModel = updates.model_name !== undefined ? updates.model_name : modelName;
    const nextTemp = updates.temperature !== undefined ? parseFloat(updates.temperature) : temperature;
    const nextAlpha = updates.hybrid_alpha !== undefined ? parseFloat(updates.hybrid_alpha) : hybridAlpha;
    const nextRerank = updates.use_reranking !== undefined ? updates.use_reranking : useReranking;
    
    const payload = {
      model_name: nextModel,
      temperature: nextTemp,
      max_tokens: 1000,
      hybrid_alpha: nextAlpha,
      hybrid_beta: parseFloat((1.0 - nextAlpha).toFixed(2)),
      use_reranking: nextRerank
    };

    try {
      const response = await fetch(`${API_BASE}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (response.ok) {
        setModelName(data.settings.model_name);
        setTemperature(data.settings.temperature);
        setHybridAlpha(data.settings.hybrid_alpha);
        setUseReranking(data.settings.use_reranking);
      } else {
        throw new Error(data.detail || 'Failed to save settings');
      }
    } catch (error) {
      console.error('Failed to save settings:', error);
    } finally {
      setIsSavingSettings(false);
    }
  };

  const fetchHealthAndDocs = async () => {
    try {
      const [healthRes, docsRes] = await Promise.all([
        fetch(`${API_BASE}/health`),
        fetch(`${API_BASE}/documents`)
      ]);
      const health = await healthRes.json();
      const docs = await docsRes.json();

      setDocuments(docs);

      if (health.initialized) {
        setStatus({ state: 'ready', text: `System Ready` });
      } else if (docs.length > 0) {
        setStatus({ state: 'pending', text: 'Needs Indexing' });
      } else {
        setStatus({ state: 'empty', text: 'No Documents' });
      }
      setSystemInfo({ size_kb: health.total_size_kb || 0, initialized: health.initialized });
    } catch (error) {
      console.error(error);
      setStatus({ state: 'error', text: 'Offline' });
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setStatus({ state: 'indexing', text: 'Uploading...' });

    const formData = new FormData();
    formData.append('file', file);

    try {
      await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      await fetchHealthAndDocs();
    } catch (error) {
      console.error('Upload failed:', error);
      setStatus({ state: 'error', text: 'Upload Failed' });
    } finally {
      setIsUploading(false);
      e.target.value = null;
    }
  };

  const handleIngest = async () => {
    setIsIndexing(true);
    setStatus({ state: 'indexing', text: 'Building Knowledge Base...' });
    try {
      await fetch(`${API_BASE}/ingest`, { method: 'POST' });
      await fetchHealthAndDocs();
    } catch (error) {
      console.error('Ingest failed:', error);
    } finally {
      setIsIndexing(false);
    }
  };

  const handleRevokeDocument = async (filename) => {
    try {
      const response = await fetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(`Server returned ${response.status}: ${errData.detail || 'Failed to delete'}`);
      }
      // If it passes, refresh local UI
      await fetchHealthAndDocs();
    } catch (error) {
      console.error('Revoke failed:', error);
      alert(`Failed to revoke document: ${error.message}\n(Hint: Did you completely restart your Python/FastAPI backend server since the new endpoint was added?)`);
    }
  };

  const handleClearDocuments = async () => {
    try {
      await fetch(`${API_BASE}/documents`, { method: 'DELETE' });
      setDocuments([]);
      setStatus({ state: 'empty', text: 'No Documents' });
    } catch (error) {
      console.error('Clear failed:', error);
    }
  };

  const handleSendMessage = async (e, directMessage) => {
    e?.preventDefault();
    const messageText = directMessage || inputMessage;
    if (!messageText.trim() || isLoading) return;

    if (view === 'welcome') setView('session');

    const userMessage = { id: Date.now(), type: 'user', content: messageText };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userMessage.content })
      });

      const data = await response.json();

      if (!response.ok) throw new Error(data.detail || 'Failed to query');

      setLastMetrics({
        latency: data.latency_ms,
        confidence: data.confidence * 100
      });

      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        type: 'ai',
        content: data.answer,
        sources: data.sources,
        latency: data.latency_ms
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        type: 'ai',
        content: `Error: ${error.message}. Please check if the system is indexed.`
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const suggestedQueries = [
    "How can I find the full path to a font from its display name on a Mac?",
    "Is there a simple way to get a preview JPEG of a PDF on Windows?",
    "What continuous integration systems are suitable for a Python codebase?",
    "How do I iterate over a result set using cx_Oracle in Python?",
    "How do you express binary literals in Python?",
    "How do you add a method to an existing object instance in Python?"
  ];

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="logo-container">
            <ShieldCheck className="logo-icon" />
            <div>
              <h1 className="brand-name">Architect AI</h1>
              <span className="brand-tag">ENTERPRISE RAG</span>
            </div>
          </div>
        </div>

        <div className="sidebar-actions">
          <button className="btn-new-session" onClick={() => { navigateTo('query-lab'); setMessages([]); }}>
            <Plus size={18} />
            <span>New Research Session</span>
          </button>
        </div>

        <nav className="sidebar-nav">
          <a href="/knowledge-base" className={`nav-item ${view === 'knowledge-base' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); navigateTo('knowledge-base'); }}>
            <Database size={20} />
            <span>Knowledge Base</span>
          </a>
          <a href="/query-lab" className={`nav-item ${(view === 'query-lab' || view === 'session' || view === 'welcome') ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); navigateTo('query-lab'); }}>
            <FlaskConical size={20} />
            <span>Query Lab</span>
          </a>
          <a href="/chat-history" className={`nav-item ${view === 'chat-history' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); navigateTo('chat-history'); }}>
            <History size={20} />
            <span>Chat History</span>
          </a>
          <a href="/analytics" className={`nav-item ${view === 'analytics' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); navigateTo('analytics'); }}>
            <BarChart3 size={20} />
            <span>Analytics</span>
          </a>
          <a href="/settings" className={`nav-item ${view === 'settings' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); navigateTo('settings'); }}>
            <Settings size={20} />
            <span>Settings</span>
          </a>
        </nav>

        <div className="sidebar-footer">
          <div className="ingestion-accordion">
            <button
              className="accordion-header"
              onClick={() => setIsIngestionOpen(!isIngestionOpen)}
            >
              <span>Document Ingestion</span>
              <ChevronDown className={`chevron ${isIngestionOpen ? 'open' : ''}`} size={16} />
            </button>

            {/* Sidebar Content wrapper for animation */}
            <div className={`accordion-content ${isIngestionOpen ? 'open' : ''}`}>
              <label className="upload-mini">
                <input
                  type="file"
                  onChange={handleFileUpload}
                  accept=".pdf,.txt,.docx"
                  style={{ display: 'none' }}
                />
                <span className="upload-text">Add new documents</span>
              </label>

              <div className="doc-list">
                {documents.length > 0 ? (
                  documents.map((doc, idx) => (
                    <div key={idx} className="doc-item">
                      <FileText size={14} className="doc-icon" />
                      <div className="doc-details">
                        <div className="doc-name-small" title={doc.name}>{doc.name}</div>
                        <div className="doc-size">{doc.size_kb} KB</div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="doc-empty">Empty knowledge base</div>
                )}
              </div>

              <div className="ingestion-item">
                <button className="btn-clear" onClick={handleClearDocuments}>Clear</button>
                <button
                  className="btn-process"
                  onClick={handleIngest}
                  disabled={documents.length === 0 || isIndexing}
                >
                  {isIndexing ? 'Indexing...' : 'Index'}
                </button>
              </div>
            </div>
          </div>

          <a href="#" className="nav-item footer-link">
            <HelpCircle size={20} />
            <span>Help Center</span>
          </a>

          <div className="profile-card">
            <div className="profile-icon">
              <ShieldCheck size={20} />
            </div>
            <div className="profile-info">
              <div className="profile-name">Enterprise Workspace</div>
              <div className="profile-tier">Premium Tier</div>
            </div>
          </div>

          <div className="system-status">
            <div className="status-indicator">
              <span className={`status-dot ${status.state}`}></span>
              <span>Backend: {status.text}</span>
            </div>
            <div className="status-meta">
              Documents: {documents.length}
            </div>
          </div>
        </div>
      </aside>

      {/* Main Area */}
      <main className="main-viewport">
        <div className="scroll-container">
          {(view === 'query-lab' || view === 'welcome' || view === 'session') && (
            <div className="welcome-screen">
            <div className="welcome-content">
              <div className="session-badges">
                <span className="badge session-id">RESEARCH SESSION #402</span>
                <span className="badge model-id">ACTIVE MODEL: GPT-40 ARCHITECT</span>
                {lastMetrics.confidence > 0 && <span className="badge model-id">QUALITY: {lastMetrics.confidence.toFixed(1)}%</span>}
              </div>

              <h1 className="session-title">Impact of Decentralized Compute on Enterprise AI Infrastructure</h1>

              <p className="welcome-text">
                The shift toward decentralized compute resources represents a fundamental pivot in how enterprise-grade RAG systems manage latency and data sovereignty. By distributing inference across edge nodes, organizations can achieve a 40% reduction in round-trip data transit times.
              </p>

              {/* Metrics Grid */}
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-value">{lastMetrics.confidence.toFixed(1)}%</div>
                  <div className="metric-label">QUALITY SCORE</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{lastMetrics.latency > 0 ? `${lastMetrics.latency}ms` : '---'}</div>
                  <div className="metric-label">RESPONSE SPEED</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{systemInfo.size_kb > 0 ? `${Math.round(systemInfo.size_kb)} KB` : '0 KB'}</div>
                  <div className="metric-label">INDEX VOLUME</div>
                </div>
              </div>

              <div className="content-section">
                <h2 className="section-header-bar">Structural Rigor and Elasticity</h2>
                <p className="welcome-text">
                  Modern architectural patterns favor a hybrid approach. While long-term knowledge retrieval is centralized in high-density vector stores, the immediate conversational context is increasingly cached at the perimeter. This "Informed Architect" strategy ensures that high-priority queries bypass the standard queue, leveraging dedicated hardware accelerators.
                </p>
              </div>

              <div className="hero-visualization">
                <img src={networkImage} alt="Topology" className="hero-img" />
                <div className="hero-badge">Network Topology Visualization</div>
              </div>

              <div className="suggested-section">
                <h3 className="section-label">SUGGESTED RESEARCH QUERIES</h3>
                <div className="query-grid">
                  {suggestedQueries.map((q, i) => (
                    <button key={i} className="query-card" onClick={() => handleSendMessage(null, q)}>
                      {q}
                      <ArrowRight size={16} />
                    </button>
                  ))}
                </div>
              </div>

              {messages.length > 0 && (
                <div className="chat-messages session-chat" style={{ marginTop: '32px', borderTop: '1px solid var(--border)', paddingTop: '40px' }}>
                  <h3 className="section-label" style={{ marginBottom: '24px' }}>ACTIVE SESSION LOG</h3>
                  {messages.map(msg => (
                    <div key={msg.id} className={`chat-message ${msg.type}`}>
                      <div className="msg-avatar">
                        {msg.type === 'user' ? <User size={20} /> : <Bot size={20} />}
                      </div>
                      <div className="msg-body">
                        <div className="msg-content">{msg.content}</div>
                        {msg.sources && (
                          <div className="msg-sources">
                            {msg.sources.map((s, i) => <span key={i} className="source-chip">{s}</span>)}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {isLoading && (
                    <div className="chat-message ai loading">
                      <div className="msg-avatar"><Loader2 className="animate-spin" size={20} /></div>
                      <div className="msg-body"><div className="loading-dots">...</div></div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>
          </div>
          )}

          {view === 'knowledge-base' && (
            <div className="screen-container">
              <div className="screen-header">
                <h1 className="screen-title">Knowledge Base</h1>
                <p className="screen-subtitle">Manage and monitor enterprise ingested documents, vector statuses, and chunking.</p>
              </div>
              <div className="data-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Document File</th>
                      <th>Size (KB)</th>
                      <th>Ingested By</th>
                      <th>Status</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {documents.length > 0 ? (
                      documents.map((doc, idx) => (
                        <tr key={idx} className="table-row">
                          <td style={{ fontWeight: 600 }}>{doc.name}</td>
                          <td>{doc.size_kb}</td>
                          <td>System Admin</td>
                          <td><span className="status-badge success"><CheckCircle2 size={12}/> Active Index</span></td>
                          <td>
                            <button className="btn-clear" style={{ padding: '6px 12px', background: 'var(--slate-100)', color: '#ef4444' }} onClick={() => handleRevokeDocument(doc.name)}>Revoke</button>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="5" style={{ textAlign: 'center', padding: '32px', color: 'var(--slate-400)' }}>
                          No documents currently mapped in the semantic database.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {view === 'chat-history' && (
            <div className="screen-container">
              <div className="screen-header">
                <h1 className="screen-title">Chat History</h1>
                <p className="screen-subtitle">Review previous AI research sessions and organizational queries.</p>
              </div>
              <div className="history-grid">
                <div className="history-card">
                  <div className="history-date">Today, 2:14 PM</div>
                  <div className="history-title">Impact of Decentralized Compute</div>
                  <div className="history-preview">The shift toward decentralized compute resources represents a fundamental pivot in how enterprise RAG systems manage latency...</div>
                </div>
                <div className="history-card">
                  <div className="history-date">Yesterday, 10:30 AM</div>
                  <div className="history-title">Company Remote Work Policy</div>
                  <div className="history-preview">According to the HR guidelines updated in 2024, employees are allowed up to 3 days of remote work per week pending manager approval...</div>
                </div>
                <div className="history-card">
                  <div className="history-date">Apr 04, 2026</div>
                  <div className="history-title">Q3 Financial Summary</div>
                  <div className="history-preview">The Q3 revenue reports indicate a 12% YoY growth primarily driven by the enterprise cloud sector...</div>
                </div>
              </div>
            </div>
          )}

          {view === 'analytics' && (
            <div className="screen-container">
              <div className="screen-header">
                <h1 className="screen-title">Traffic & Usage Analytics</h1>
                <p className="screen-subtitle">Monitor token consumption, average latencies, and retrieval accuracy.</p>
              </div>
              <div className="metrics-grid" style={{ marginBottom: '40px' }}>
                <div className="metric-card">
                  <div className="metric-value">42.8k</div>
                  <div className="metric-label">TOTAL TOKENS GENERATED</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">99.9%</div>
                  <div className="metric-label">SYSTEM UPTIME</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">68</div>
                  <div className="metric-label">ACTIVE DAILY USERS</div>
                </div>
              </div>
              <div className="settings-section">
                <h3 className="settings-section-title">Performance Vitals</h3>
                <div className="setting-row">
                  <div className="setting-info">
                    <div className="setting-name">Vector Search Latency</div>
                    <div className="setting-desc">Average time required to pull k-nearest neighbors.</div>
                  </div>
                  <div><span style={{ fontWeight: 800, color: 'var(--accent)' }}>48ms</span></div>
                </div>
                <div className="setting-row">
                  <div className="setting-info">
                    <div className="setting-name">Context Window Utilization</div>
                    <div className="setting-desc">Average percentage of available LLM context used per query.</div>
                  </div>
                  <div><span style={{ fontWeight: 800, color: 'var(--text-main)' }}>64%</span></div>
                </div>
              </div>
            </div>
          )}

          {view === 'settings' && (
            <div className="screen-container">
              <div className="screen-header">
                <h1 className="screen-title">System Settings</h1>
                <p className="screen-subtitle">Adjust core LLM parameters, search weights, and caching behaviors in real-time.</p>
              </div>
              <div className="settings-section">
                <h3 className="settings-section-title">Model Configuration</h3>
                <div className="setting-row">
                  <div className="setting-info">
                    <div className="setting-name">Active Engine</div>
                    <div className="setting-desc">Select the primary LLM used for inference and summarization.</div>
                  </div>
                  <select 
                    className="settings-select" 
                    value={modelName}
                    onChange={(e) => {
                      setModelName(e.target.value);
                      handleSaveSettings({ model_name: e.target.value });
                    }}
                  >
                    <option value="openrouter/free">Auto-Select Free Engine (Free)</option>
                    <option value="openai/gpt-4o">GPT-4o Architect (Paid)</option>
                    <option value="anthropic/claude-3-5-sonnet">Claude Enterprise (Paid)</option>
                  </select>
                </div>
                <div className="setting-row">
                  <div className="setting-info">
                    <div className="setting-name">Temperature</div>
                    <div className="setting-desc">Controls the creativity of the responses. Lower values are more deterministic.</div>
                  </div>
                  <select 
                    className="settings-select"
                    value={temperature}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      setTemperature(val);
                      handleSaveSettings({ temperature: val });
                    }}
                  >
                    <option value="0.0">0.0 (Precise)</option>
                    <option value="0.3">0.3 (Balanced)</option>
                    <option value="0.7">0.7 (Creative)</option>
                  </select>
                </div>
              </div>

              <div className="settings-section">
                <h3 className="settings-section-title">RAG Retrieval Optimization</h3>
                <div className="setting-row">
                  <div className="setting-info">
                    <div className="setting-name">Hybrid Weight (Alpha)</div>
                    <div className="setting-desc">
                      Balance Keyword Search ({Math.round(hybridAlpha * 100)}%) and Semantic Search ({Math.round((1 - hybridAlpha) * 100)}%).
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', width: '200px' }}>
                    <input 
                      type="range" 
                      min="0" 
                      max="1" 
                      step="0.05"
                      value={hybridAlpha}
                      onChange={(e) => setHybridAlpha(parseFloat(e.target.value))}
                      onMouseUp={(e) => handleSaveSettings({ hybrid_alpha: parseFloat(e.target.value) })}
                      onTouchEnd={(e) => handleSaveSettings({ hybrid_alpha: parseFloat(e.target.value) })}
                      style={{ flex: 1, cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '14px', fontWeight: 'bold', width: '32px', textAlign: 'right' }}>{hybridAlpha.toFixed(2)}</span>
                  </div>
                </div>
                
                <div className="setting-row">
                  <div className="setting-info">
                    <div className="setting-name">FlashRank Re-ranking</div>
                    <div className="setting-desc">Enable lightweight cross-encoder model to re-order hybrid documents for top accuracy.</div>
                  </div>
                  <label className="toggle-switch">
                    <input 
                      type="checkbox" 
                      checked={useReranking}
                      onChange={(e) => {
                        setUseReranking(e.target.checked);
                        handleSaveSettings({ use_reranking: e.target.checked });
                      }}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>

              <div className="settings-section">
                <h3 className="settings-section-title">Security & Guardrails</h3>
                <div className="setting-row">
                  <div className="setting-info">
                    <div className="setting-name">Strict Content Filtering</div>
                    <div className="setting-desc">Automatically blocks outputs that may violate organizational compliance policies.</div>
                  </div>
                  <label className="toggle-switch">
                    <input type="checkbox" defaultChecked />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
                <div className="setting-row">
                  <div className="setting-info">
                    <div className="setting-name">Semantic Query Caching</div>
                    <div className="setting-desc">Cache identical or semantically similar queries to save inference costs.</div>
                  </div>
                  <label className="toggle-switch">
                    <input type="checkbox" defaultChecked />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
          )}
          <div className="viewport-spacer" style={{ minHeight: '320px', flexShrink: 0 }}></div>
        </div>

        {/* Floating Input Dock */}
        {(view === 'query-lab' || view === 'welcome' || view === 'session') && (
          <div className="input-dock">
            <form className="input-field" onSubmit={handleSendMessage}>
              <textarea
                placeholder="Ask a sophisticated follow-up..."
                value={inputMessage}
                onChange={(e) => {
                  setInputMessage(e.target.value);
                  // Auto-expand textarea
                  e.target.style.height = 'auto';
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                    e.target.style.height = 'auto';
                  }
                }}
                rows="1"
              />
              <button type="submit" disabled={!inputMessage.trim() || isLoading} className="send-circle">
                {isLoading ? <Loader2 className="animate-spin" size={18} /> : <Send size={18} />}
              </button>
            </form>
          </div>
        )}
      </main>
    </div>
  );
}


export default App;

