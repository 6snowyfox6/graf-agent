import { Handle, Position } from '@xyflow/react';
import { NodeResizer, NodeToolbar } from '@xyflow/react';

export default function CustomNode({ data, selected }) {
  // data = { label, kind, width, height, color }
  const kind = data.kind || 'block';
  const customStyle = {};
  if (data.color) {
    customStyle.backgroundColor = `${data.color}22`;
    customStyle.borderColor = data.color;
    customStyle.boxShadow = selected ? `0 0 0 2px ${data.color}` : undefined;
  }
  
  const colors = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#a8a29e'];

  return (
    <>
      <NodeResizer 
        color="#6366f1" 
        isVisible={selected} 
        minWidth={120} 
        minHeight={50} 
      />
      <NodeToolbar isVisible={selected} position={Position.Top}>
        <div style={{ display: 'flex', gap: '8px', padding: '6px', background: 'rgba(25,28,36,0.9)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)' }}>
          {colors.map(c => (
            <button 
              key={c}
              style={{ width: '20px', height: '20px', borderRadius: '50%', background: c, border: 'none', cursor: 'pointer', padding: 0 }}
              onClick={() => data.onColorChange(data.id, c)}
            />
          ))}
        </div>
      </NodeToolbar>
      
      <div className={`custom-node node-${kind} ${selected ? 'selected' : ''}`} style={customStyle}>
        <Handle type="target" position={Position.Top} />
        <div className="node-badge" style={data.color ? { color: data.color } : {}}>{kind}</div>
        <div
          className="nodrag custom-node-input"
          contentEditable
          suppressContentEditableWarning
          onBlur={(e) => data.onChange(data.id, e.target.textContent)}
          style={{ cursor: 'text' }}
        >
          {data.label}
        </div>
        <Handle type="source" position={Position.Bottom} />
      </div>
    </>
  );
}
