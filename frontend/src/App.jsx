import React, { useState, useCallback, useMemo, useEffect } from 'react';
import {
  ReactFlow, MiniMap, Controls, Background, useNodesState, useEdgesState,
  addEdge, applyNodeChanges, applyEdgeChanges,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Play, Sparkles, Loader2, Download, Image as ImageIcon, LayoutTemplate, Undo2, Redo2, Moon, Sun, ChevronDown, ChevronRight, MessageSquareWarning } from 'lucide-react';
import CustomNode from './CustomNode';
import CustomEdge from './CustomEdge';
import { getLayoutedElements } from './layout';

const API_BASE = 'http://localhost:8080/api'; 

export default function App() {
  const [activeTab, setActiveTab] = useState('interactive');
  const [theme, setTheme] = useState('dark');

  // --- INTERACTIVE CANVAS STATE ---
  const [nodes, setNodes] = useNodesState([]);
  const [edges, setEdges] = useEdgesState([]);
  const [interactivePrompt, setInteractivePrompt] = useState('Сделай архитектуру микросервисов для интернет магазина');
  const [userFeedback, setUserFeedback] = useState('');
  const [interactiveLoading, setInteractiveLoading] = useState(false);
  const [improving, setImproving] = useState(false);

  // History State
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // SHAP State
  const [shapResult, setShapResult] = useState(null);
  const [shapExpanded, setShapExpanded] = useState(false);

  // --- PLOT NEURAL NET STATE ---
  const [pnnPrompt, setPnnPrompt] = useState('Build a ResNet architecture');
  const [pnnLoading, setPnnLoading] = useState(false);
  const [pnnResult, setPnnResult] = useState(null);

  useEffect(() => {
    if (theme === 'light') {
      document.body.classList.add('light-theme');
    } else {
      document.body.classList.remove('light-theme');
    }
  }, [theme]);

  const pushHistory = useCallback((newNodes, newEdges) => {
    setHistory(prev => {
        const nextHist = prev.slice(0, historyIndex + 1);
        nextHist.push({ nodes: JSON.parse(JSON.stringify(newNodes)), edges: JSON.parse(JSON.stringify(newEdges)) });
        return nextHist;
    });
    setHistoryIndex(prev => prev + 1);
  }, [historyIndex]);

  const undo = () => {
    if (historyIndex > 0) {
      setHistoryIndex(prev => prev - 1);
      const state = history[historyIndex - 1];
      setNodes(state.nodes);
      setEdges(state.edges);
    }
  };

  const redo = () => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(prev => prev + 1);
      const state = history[historyIndex + 1];
      setNodes(state.nodes);
      setEdges(state.edges);
    }
  };

  const handleNodeLabelChange = useCallback((id, newLabel) => {
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === id) n.data = { ...n.data, label: newLabel };
        return n;
      })
    );
  }, []);

  const handleNodeColorChange = useCallback((id, newColor) => {
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === id) n.data = { ...n.data, color: newColor };
        return n;
      })
    );
  }, []);

  const handleEdgeDelete = useCallback((id) => {
    setEdges((eds) => eds.filter((e) => e.id !== id));
  }, [setEdges]);

  const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);
  const edgeTypes = useMemo(() => ({ custom: CustomEdge }), []);

  const onNodesChange = useCallback((changes) => setNodes((nds) => applyNodeChanges(changes, nds)), []);
  const onEdgesChange = useCallback((changes) => setEdges((eds) => applyEdgeChanges(changes, eds)), []);
  const onConnect = useCallback((params) => {
    setEdges((eds) => addEdge({ ...params, type: 'custom', animated: true, data: { onDelete: handleEdgeDelete } }, eds));
  }, [handleEdgeDelete]);

  const onNodeDragStop = useCallback((event, node, newNodes) => {
    pushHistory(newNodes, edges);
  }, [edges, pushHistory]);

  const transformBackendToFlow = (backendGraph) => {
    const rawNodes = (backendGraph.nodes || []).map((n) => ({
      id: n.id,
      type: 'custom',
      position: { x: 0, y: 0 },
      data: { 
        label: n.label, 
        kind: n.kind,
        color: n.color,
        onChange: handleNodeLabelChange,
        onColorChange: handleNodeColorChange
      },
    }));

    const rawEdges = (backendGraph.edges || []).map((e) => ({
      id: `e-${e.source}-${e.target}`,
      source: e.source,
      target: e.target,
      label: e.label || '',
      type: 'custom',
      animated: true,
      data: { onDelete: handleEdgeDelete }
    }));

    return getLayoutedElements(rawNodes, rawEdges);
  };

  const transformFlowToBackend = () => {
    return {
      nodes: nodes.map(n => ({ id: n.id, label: n.data.label, kind: n.data.kind, color: n.data.color })),
      edges: edges.map(e => ({ source: e.source, target: e.target, label: e.label }))
    };
  };

  const generateInteractive = async () => {
    if (!interactivePrompt.trim()) return;
    setInteractiveLoading(true);
    setShapResult(null);
    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: interactivePrompt }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        const { nodes: layoutedNodes, edges: layoutedEdges } = transformBackendToFlow(data.graph);
        setNodes(layoutedNodes);
        setEdges(layoutedEdges);
        pushHistory(layoutedNodes, layoutedEdges);
      }
    } catch (err) {
      console.error(err);
      alert('Error fetching interactive response');
    } finally {
      setInteractiveLoading(false);
    }
  };

  const improveInteractive = async () => {
    setImproving(true);
    try {
      const res = await fetch(`${API_BASE}/improve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: interactivePrompt, graph: transformFlowToBackend(), user_feedback: userFeedback }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        const { nodes: layoutedNodes, edges: layoutedEdges } = transformBackendToFlow(data.graph);
        setNodes(layoutedNodes);
        setEdges(layoutedEdges);
        pushHistory(layoutedNodes, layoutedEdges);
        
        if (data.shap) {
          setShapResult(data.shap);
          setShapExpanded(true);
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setImproving(false);
    }
  };

  const handleExportInteractive = async () => {
    try {
      const graph = transformFlowToBackend();
      const res = await fetch(`${API_BASE}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(graph),
      });
      
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `diagram_export_${Date.now()}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export failed', err);
      alert('Failed to export diagram.');
    }
  };

  const generatePNN = async () => {
    if (!pnnPrompt.trim()) return;
    setPnnLoading(true);
    setPnnResult(null);
    try {
      const res = await fetch(`${API_BASE}/generate_pnn`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: pnnPrompt }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        setPnnResult(data);
      }
    } catch (err) {
      console.error(err);
      alert('Error fetching PlotNeuralNet response');
    } finally {
      setPnnLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="sidebar" style={{ overflowY: 'auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <h1 className="title" style={{ margin: 0 }}>Graf Center</h1>
          <button 
            onClick={() => setTheme(t => t === 'light' ? 'dark' : 'light')}
            style={{ width: '32px', height: '32px', padding: 0, borderRadius: '50%', background: 'transparent', border: '1px solid var(--panel-border)', color: 'var(--text-color)', marginBottom: 0 }}
          >
            {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
          </button>
        </div>
        
        <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
          <button 
            className={`tab-btn ${activeTab === 'interactive' ? 'active' : ''}`}
            onClick={() => setActiveTab('interactive')}
            style={{ flex: 1, padding: '8px', fontSize: '12px', minWidth: '130px' }}
          >
            <LayoutTemplate size={14} /> Interactive
          </button>
          <button 
            className={`tab-btn ${activeTab === 'plotneuralnet' ? 'active' : ''}`}
            onClick={() => setActiveTab('plotneuralnet')}
            style={{ flex: 1, padding: '8px', fontSize: '12px', minWidth: '130px' }}
          >
            <ImageIcon size={14} /> 3D PNN
          </button>
        </div>

        {activeTab === 'interactive' && (
          <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            
            <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
              <button className="secondary" onClick={undo} disabled={historyIndex <= 0} style={{ padding: '8px', flex: 1 }}>
                <Undo2 size={16} />
              </button>
              <button className="secondary" onClick={redo} disabled={historyIndex >= history.length - 1} style={{ padding: '8px', flex: 1 }}>
                <Redo2 size={16} />
              </button>
            </div>

            <textarea 
              placeholder="Опиши архитектуру системы..."
              value={interactivePrompt}
              onChange={(e) => setInteractivePrompt(e.target.value)}
              style={{ minHeight: '80px' }}
            />
            
            <button onClick={generateInteractive} disabled={interactiveLoading || improving}>
              {interactiveLoading ? <Loader2 className="spinner" size={18} /> : <Play size={18} />}
              Сгенерировать
            </button>
            
            <div style={{ marginTop: '16px', background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', border: '1px solid var(--panel-border)' }}>
               <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', fontSize: '12px', color: 'var(--text-color)', opacity: 0.8 }}>
                 <MessageSquareWarning size={14} /> Мой комментарий (опционально)
               </div>
               <textarea 
                  placeholder="Что именно исправить? (например: удали базу данных)"
                  value={userFeedback}
                  onChange={(e) => setUserFeedback(e.target.value)}
                  style={{ minHeight: '50px', marginBottom: '12px' }}
                />
              <button className="secondary" onClick={improveInteractive} disabled={interactiveLoading || improving || nodes.length === 0} style={{ marginBottom: 0 }}>
                {improving ? <Loader2 className="spinner" size={18} /> : <Sparkles size={18} />}
                AI Critic (Подчинить)
              </button>
            </div>

            {shapResult && (
               <div className="shap-container">
                  <div className="shap-header" onClick={() => setShapExpanded(!shapExpanded)}>
                    <span>SHAP Interpretability</span>
                    {shapExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  </div>
                  {shapExpanded && (
                    <div className="shap-content">
                       {shapResult.bar && (
                         <div>
                           <div style={{fontSize:'11px', marginBottom:'4px', opacity:0.7}}>Feature Importance (Bar)</div>
                           <img src={`http://localhost:8080${shapResult.bar}`} alt="SHAP Bar" className="shap-img" />
                         </div>
                       )}
                       {shapResult.network && (
                         <div>
                           <div style={{fontSize:'11px', marginBottom:'4px', opacity:0.7}}>Concept Traceability (Network)</div>
                           <img src={`http://localhost:8080${shapResult.network}`} alt="SHAP Network" className="shap-img" />
                         </div>
                       )}
                    </div>
                  )}
               </div>
            )}

            <div style={{ marginTop: 'auto', paddingTop: '24px' }}>
              <button className="secondary" onClick={handleExportInteractive} disabled={nodes.length === 0}>
                <Download size={18} />
                Сохранить схему (ZIP)
              </button>
            </div>
          </div>
        )}

        {activeTab === 'plotneuralnet' && (
          <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <textarea 
              placeholder="Describe the neural network architecture (e.g. U-Net with ResNet34 backbone)..."
              value={pnnPrompt}
              onChange={(e) => setPnnPrompt(e.target.value)}
            />
            
            <button onClick={generatePNN} disabled={pnnLoading}>
              {pnnLoading ? <Loader2 className="spinner" size={18} /> : <Play size={18} />}
              Сгенерировать 3D
            </button>
            
            {pnnResult && pnnResult.pdf_url && (
              <a href={`http://localhost:8080${pnnResult.pdf_url}`} target="_blank" rel="noopener noreferrer" style={{textDecoration:'none'}}>
                <button className="secondary" style={{ marginTop: '16px' }}>
                  <Download size={18} /> Скачать PDF
                </button>
              </a>
            )}
            
            {pnnResult && pnnResult.png_url && (
              <a href={`http://localhost:8080${pnnResult.png_url}`} target="_blank" rel="noopener noreferrer" style={{textDecoration:'none'}}>
                <button className="secondary" style={{ marginTop: '6px' }}>
                  <Download size={18} /> Скачать PNG
                </button>
              </a>
            )}
          </div>
        )}
      </div>
      
      <div className="canvas-container" style={{ background: 'transparent' }}>
        {activeTab === 'interactive' ? (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeDragStop={onNodeDragStop}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            attributionPosition="bottom-right"
          >
            <Background color={theme === 'light' ? '#000' : '#fff'} gap={16} size={1} opacity={0.05} />
            <Controls />
            <MiniMap 
              nodeColor={(n) => n.data?.color || (theme === 'light' ? '#e2e8f0' : '#3b82f6')}
              maskColor={theme === 'light' ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0.4)'}
              style={{ backgroundColor: theme === 'light' ? '#fff' : '#1e2029' }}
            />
          </ReactFlow>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', padding: '40px' }}>
            {pnnLoading && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', color: 'var(--text-color)' }}>
                <Loader2 className="spinner" size={48} style={{ marginBottom: '16px' }} />
                <h3>Отрисовка 3D сети (включает вызов LaTeX компилятора)...</h3>
              </div>
            )}
            {!pnnLoading && pnnResult?.png_url && (
                <div style={{ padding: '24px', background: 'white', borderRadius: '16px', boxShadow: '0 10px 40px rgba(0,0,0,0.3)' }}>
                  <img src={`http://localhost:8080${pnnResult.png_url}`} alt="PlotNeuralNet Result" style={{ maxWidth: '100%', maxHeight: '70vh', objectFit: 'contain' }} />
                </div>
            )}
            {!pnnLoading && !pnnResult && (
               <div style={{ color: 'var(--text-color)', opacity: 0.3, fontSize: '18px' }}>
                 Введите промпт нейросети в панели слева
               </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
