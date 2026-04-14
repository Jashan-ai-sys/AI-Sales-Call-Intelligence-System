/**
 * SalesIQ — AI Sales Call Intelligence Dashboard
 * Frontend application logic
 */

const API_BASE = window.location.origin;

// ─── State ───────────────────────────────────────────────
let currentView = 'dashboard';
let callsData = [];

// ─── Initialization ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUploadZone();
    checkAPIHealth();
    loadDashboard();
});

// ─── Navigation ──────────────────────────────────────────
function initNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const view = item.dataset.view;
            if (view) showView(view);
        });
    });
}

function showView(viewName) {
    // Hide all views
    document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
    
    // Show target view
    const target = document.getElementById(`view-${viewName}`);
    if (target) {
        target.classList.remove('hidden');
    }
    
    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });
    
    currentView = viewName;
    
    // Load data for view
    if (viewName === 'dashboard') loadDashboard();
    if (viewName === 'calls') loadCallHistory();
}

// ─── API Health Check ────────────────────────────────────
async function checkAPIHealth() {
    const statusEl = document.getElementById('api-status');
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        if (res.ok) {
            statusEl.className = 'api-status connected';
            statusEl.querySelector('span').textContent = 'System Online';
        } else {
            throw new Error('Not healthy');
        }
    } catch (e) {
        statusEl.className = 'api-status error';
        statusEl.querySelector('span').textContent = 'API Offline';
    }
}

// ─── Dashboard ───────────────────────────────────────────
async function loadDashboard() {
    try {
        const res = await fetch(`${API_BASE}/api/stats`);
        if (!res.ok) return;
        
        const stats = await res.json();
        
        // Update KPIs
        animateCounter('kpi-total-calls', stats.total_calls);
        document.getElementById('kpi-avg-score').textContent = stats.total_calls > 0 ? `${stats.avg_score}` : '—';
        document.getElementById('kpi-avg-sentiment').textContent = stats.total_calls > 0 ? `${(stats.avg_sentiment * 100).toFixed(0)}%` : '—';
        document.getElementById('kpi-conversion').textContent = stats.total_calls > 0 ? `${(stats.avg_conversion_probability * 100).toFixed(0)}%` : '—';
        
        // Objections chart
        if (stats.top_objections && stats.top_objections.length > 0) {
            renderObjectionsChart(stats.top_objections);
        }
        
        // Score distribution
        if (stats.score_distribution) {
            renderScoreDistribution(stats.score_distribution);
        }
        
        // Recent calls table
        if (stats.recent_calls && stats.recent_calls.length > 0) {
            renderRecentCalls(stats.recent_calls);
        }
    } catch (e) {
        console.error('Failed to load dashboard:', e);
    }
}

function animateCounter(elementId, targetValue) {
    const el = document.getElementById(elementId);
    const duration = 800;
    const start = parseInt(el.textContent) || 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = Math.round(start + (targetValue - start) * eased);
        el.textContent = current;
        if (progress < 1) requestAnimationFrame(update);
    }
    
    requestAnimationFrame(update);
}

function renderObjectionsChart(objections) {
    const container = document.getElementById('objections-chart');
    const maxCount = Math.max(...objections.map(o => o.count));
    
    let html = '<div class="objection-bar-group">';
    objections.forEach(obj => {
        const width = (obj.count / maxCount) * 100;
        html += `
            <div class="objection-item">
                <span class="objection-label">${obj.category}</span>
                <div class="objection-bar-wrapper">
                    <div class="objection-bar ${obj.category}" style="width: ${width}%"></div>
                </div>
                <span class="objection-count">${obj.count}</span>
            </div>
        `;
    });
    html += '</div>';
    container.innerHTML = html;
    
    // Animate bars
    setTimeout(() => {
        container.querySelectorAll('.objection-bar').forEach(bar => {
            bar.style.width = bar.style.width; // trigger reflow
        });
    }, 50);
}

function renderScoreDistribution(dist) {
    const container = document.getElementById('score-dist-chart');
    const maxVal = Math.max(dist.excellent, dist.good, dist.average, dist.poor, 1);
    
    const categories = [
        { key: 'excellent', label: '80-100' },
        { key: 'good', label: '60-79' },
        { key: 'average', label: '40-59' },
        { key: 'poor', label: '0-39' }
    ];
    
    let html = '<div class="score-dist-group">';
    categories.forEach(cat => {
        const height = Math.max(8, (dist[cat.key] / maxVal) * 160);
        html += `
            <div class="score-bar-container">
                <span class="score-bar-value">${dist[cat.key]}</span>
                <div class="score-bar-visual ${cat.key}" style="height: ${height}px"></div>
                <span class="score-bar-label">${cat.label}</span>
            </div>
        `;
    });
    html += '</div>';
    container.innerHTML = html;
}

function renderRecentCalls(calls) {
    const container = document.getElementById('recent-calls-table');
    
    let html = `
        <table class="calls-table">
            <thead>
                <tr>
                    <th>Call ID</th>
                    <th>File</th>
                    <th>Score</th>
                    <th>Sentiment</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    calls.forEach(call => {
        const scoreClass = getScoreClass(call.call_score);
        const time = new Date(call.timestamp).toLocaleString();
        html += `
            <tr onclick="viewCallDetail('${call.id}')">
                <td style="font-family: var(--font-mono, monospace); color: var(--accent-primary);">${call.id}</td>
                <td>${call.filename}</td>
                <td><span class="score-badge ${scoreClass}">${call.call_score}</span></td>
                <td><span class="sentiment-badge ${call.overall_sentiment}">${getSentimentEmoji(call.overall_sentiment)} ${call.overall_sentiment}</span></td>
                <td style="color: var(--text-muted); font-size: 0.82rem;">${time}</td>
            </tr>
        `;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

// ─── Upload Zone ─────────────────────────────────────────
function initUploadZone() {
    const zone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    
    // Drag and drop
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });
    
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) uploadFile(files[0]);
    });
    
    // Click to upload
    zone.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') {
            fileInput.click();
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });
}

async function uploadFile(file) {
    const zone = document.getElementById('upload-zone');
    const panel = document.getElementById('processing-panel');
    
    zone.style.display = 'none';
    panel.classList.remove('hidden');
    
    // Animate pipeline steps
    const steps = ['step-upload', 'step-transcribe', 'step-nlp', 'step-moments', 'step-llm', 'step-rag'];
    let currentStep = 0;
    
    function advanceStep() {
        if (currentStep > 0) {
            document.getElementById(steps[currentStep - 1]).classList.remove('active');
            document.getElementById(steps[currentStep - 1]).classList.add('done');
        }
        if (currentStep < steps.length) {
            document.getElementById(steps[currentStep]).classList.add('active');
            currentStep++;
        }
    }
    
    advanceStep(); // Upload
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Simulate step progression during actual upload
        const stepInterval = setInterval(() => {
            if (currentStep < steps.length) advanceStep();
        }, 2000);
        
        const res = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        clearInterval(stepInterval);
        
        // Mark all steps as done
        steps.forEach(s => {
            document.getElementById(s).classList.remove('active');
            document.getElementById(s).classList.add('done');
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }
        
        const result = await res.json();
        
        document.getElementById('processing-title').textContent = '✅ Analysis Complete!';
        showToast(`Call analyzed! Score: ${result.call_score}/100`, 'success');
        
        // Reset after delay and show result
        setTimeout(() => {
            panel.classList.add('hidden');
            zone.style.display = '';
            steps.forEach(s => {
                document.getElementById(s).classList.remove('active', 'done');
            });
            document.getElementById('processing-title').textContent = 'Processing...';
            
            viewCallDetail(result.id, result);
        }, 1500);
        
    } catch (e) {
        document.getElementById('processing-title').textContent = '❌ Analysis Failed';
        showToast(e.message, 'error');
        
        setTimeout(() => {
            panel.classList.add('hidden');
            zone.style.display = '';
            steps.forEach(s => {
                document.getElementById(s).classList.remove('active', 'done');
            });
            document.getElementById('processing-title').textContent = 'Processing...';
        }, 3000);
    }
}

// ─── Demo ────────────────────────────────────────────────
async function runDemo() {
    const btn = document.getElementById('btn-run-demo');
    btn.disabled = true;
    btn.innerHTML = `<div class="processing-spinner" style="width:16px;height:16px;border-width:2px;"></div> Running Demo...`;
    
    try {
        const res = await fetch(`${API_BASE}/api/demo`, { method: 'POST' });
        if (!res.ok) throw new Error('Demo failed');
        
        const result = await res.json();
        showToast(`Demo analyzed! Score: ${result.call_score}/100`, 'success');
        
        // Refresh dashboard
        loadDashboard();
        
        // Show result
        setTimeout(() => {
            viewCallDetail(result.id, result);
        }, 500);
        
    } catch (e) {
        showToast('Demo failed: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Demo Analysis`;
    }
}

// ─── Call History ────────────────────────────────────────
async function loadCallHistory() {
    const container = document.getElementById('calls-list');
    
    try {
        const res = await fetch(`${API_BASE}/api/calls`);
        if (!res.ok) return;
        
        const data = await res.json();
        callsData = data.calls;
        
        if (callsData.length === 0) {
            container.innerHTML = `
                <div class="empty-state" style="grid-column: 1/-1;">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
                    <h3>No calls yet</h3>
                    <p>Upload a sales call or run a demo to get started</p>
                </div>
            `;
            return;
        }
        
        let html = '';
        callsData.forEach(call => {
            const scoreClass = getScoreClass(call.call_score);
            const time = new Date(call.timestamp).toLocaleString();
            html += `
                <div class="call-card" onclick="viewCallDetail('${call.id}')">
                    <div class="call-card-header">
                        <span class="call-card-title">${call.filename}</span>
                        <span class="score-badge ${scoreClass}">${call.call_score}</span>
                    </div>
                    <div class="call-card-meta">
                        <div class="call-meta-row">
                            <span>Sentiment</span>
                            <span class="sentiment-badge ${call.overall_sentiment.label}">
                                ${getSentimentEmoji(call.overall_sentiment.label)} ${call.overall_sentiment.label}
                            </span>
                        </div>
                        <div class="call-meta-row">
                            <span>Conversion</span>
                            <span>${(call.conversion_probability * 100).toFixed(0)}%</span>
                        </div>
                        <div class="call-meta-row">
                            <span>Objections</span>
                            <span>${call.objection_count}</span>
                        </div>
                    </div>
                    <div class="call-card-footer">
                        <span class="tag">${call.id}</span>
                        <span class="tag">${call.magic_moment_count} moments</span>
                    </div>
                </div>
            `;
        });
        container.innerHTML = html;
        
    } catch (e) {
        console.error('Failed to load calls:', e);
    }
}

// ─── Call Detail ─────────────────────────────────────────
async function viewCallDetail(callId, preloadedData = null) {
    showView('call-detail');
    const container = document.getElementById('call-detail-content');
    container.innerHTML = '<div class="shimmer" style="height:400px;border-radius:16px;"></div>';
    
    let data = preloadedData;
    if (!data) {
        try {
            const res = await fetch(`${API_BASE}/api/calls/${callId}`);
            if (!res.ok) throw new Error('Call not found');
            data = await res.json();
        } catch (e) {
            container.innerHTML = `<div class="empty-state"><h3>Call not found</h3><p>${e.message}</p></div>`;
            return;
        }
    }
    
    document.getElementById('detail-title').textContent = `Analysis: ${data.filename}`;
    
    const scoreClass = getScoreClass(data.call_score);
    const scoreColor = getScoreColor(data.call_score);
    const circumference = 2 * Math.PI * 48;
    const offset = circumference - (data.call_score / 100) * circumference;
    
    // Conversion probability color
    const convColor = data.conversion_probability > 0.6 ? 'var(--success)' : data.conversion_probability > 0.3 ? 'var(--warning)' : 'var(--danger)';
    
    let html = `
        <!-- Row 1: Score + Summary -->
        <div class="detail-grid">
            <div class="detail-card">
                <h3>📊 Call Score</h3>
                <div class="score-ring-container">
                    <div class="score-ring">
                        <svg width="120" height="120" viewBox="0 0 120 120">
                            <circle class="score-ring-bg" cx="60" cy="60" r="48"/>
                            <circle class="score-ring-fill" cx="60" cy="60" r="48" 
                                stroke="${scoreColor}" 
                                stroke-dasharray="${circumference}" 
                                stroke-dashoffset="${offset}"/>
                        </svg>
                        <span class="score-ring-value" style="color: ${scoreColor}">${data.call_score}</span>
                    </div>
                    <div class="score-breakdown-list">
                        ${Object.entries(data.score_breakdown).map(([key, val]) => `
                            <div class="breakdown-item">
                                <span class="breakdown-label">${key.replace('_', ' ')}</span>
                                <div class="breakdown-bar-wrapper">
                                    <div class="breakdown-bar" style="width: ${(val / 25) * 100}%"></div>
                                </div>
                                <span class="breakdown-value">${val}/25</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <!-- Conversion Probability -->
                <div class="conversion-meter" style="margin-top: 20px;">
                    <h3 style="font-size: 0.85rem; margin-bottom: 10px;">🎯 Conversion Probability</h3>
                    <div class="conversion-bar-wrapper">
                        <div class="conversion-bar" style="width: ${data.conversion_probability * 100}%; background: ${convColor};"></div>
                    </div>
                    <div class="conversion-label">
                        <span>Low</span>
                        <span style="font-weight:700; color: ${convColor};">${(data.conversion_probability * 100).toFixed(0)}%</span>
                        <span>High</span>
                    </div>
                </div>
            </div>
            
            <div class="detail-card">
                <h3>📝 AI Summary</h3>
                <p style="color: var(--text-secondary); line-height: 1.7; font-size: 0.92rem; margin-bottom: 20px;">
                    ${data.summary}
                </p>
                
                <h3>😊 Overall Sentiment</h3>
                <span class="sentiment-badge ${data.overall_sentiment.label}" style="font-size: 0.9rem; padding: 6px 14px;">
                    ${getSentimentEmoji(data.overall_sentiment.label)} ${data.overall_sentiment.label} (${(data.overall_sentiment.score * 100).toFixed(0)}%)
                </span>
                
                <!-- Sentiment Trajectory -->
                <div style="margin-top: 20px;">
                    <h3 style="font-size: 0.85rem; margin-bottom: 10px;">📈 Sentiment Trajectory</h3>
                    <div class="trajectory-visual">
                        ${data.sentiment_trajectory.map(s => {
                            const height = Math.max(8, Math.abs(s.score) * 50);
                            const color = s.score > 0 ? 'var(--success)' : s.score < -0.1 ? 'var(--danger)' : 'var(--text-muted)';
                            return `<div class="trajectory-bar" style="height: ${height}px; background: ${color}; opacity: 0.7;" title="${s.speaker}: ${s.score.toFixed(2)}"></div>`;
                        }).join('')}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Row 2: Suggestions + Magic Moments -->
        <div class="detail-grid">
            <div class="detail-card">
                <h3>💡 Agent Coaching Suggestions</h3>
                <div class="suggestion-list">
                    ${data.agent_suggestions.map(s => `
                        <div class="suggestion-item">
                            <span class="suggestion-icon">→</span>
                            <span>${s}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <div class="detail-card">
                <h3>⚡ Magic Moments (${data.magic_moments.length})</h3>
                ${data.magic_moments.length > 0 ? data.magic_moments.slice(0, 5).map(m => `
                    <div class="moment-item ${m.moment_type.includes('positive') ? 'positive' : 'negative'}">
                        <div class="moment-type">${m.moment_type.includes('positive') ? '✨ Positive' : '⚠️ Negative'} Turning Point</div>
                        <div class="moment-text">"${m.text.substring(0, 120)}${m.text.length > 120 ? '...' : ''}"</div>
                    </div>
                `).join('') : '<div class="empty-state-mini">No magic moments detected</div>'}
            </div>
        </div>
        
        <!-- Row 3: Objections + Entities -->
        <div class="detail-grid">
            <div class="detail-card">
                <h3>🚧 Objections Detected (${data.all_objections.length})</h3>
                ${data.all_objections.length > 0 ? `
                    <div class="objection-tags">
                        ${[...new Set(data.all_objections.map(o => o.category))].map(cat => `
                            <span class="objection-tag ${cat}">${cat} (${data.all_objections.filter(o => o.category === cat).length})</span>
                        `).join('')}
                    </div>
                    <div style="margin-top: 16px;">
                        ${data.all_objections.slice(0, 3).map(o => `
                            <div style="padding: 8px 0; border-bottom: 1px solid var(--border-subtle); font-size: 0.85rem; color: var(--text-secondary);">
                                <strong style="color: var(--text-primary);">${o.category}:</strong> "${o.text.substring(0, 100)}..."
                            </div>
                        `).join('')}
                    </div>
                ` : '<div class="empty-state-mini">No objections detected</div>'}
            </div>
            
            <div class="detail-card">
                <h3>🏷️ Entities Extracted (${data.all_entities.length})</h3>
                ${data.all_entities.length > 0 ? `
                    <div class="entity-tags">
                        ${data.all_entities.map(e => `
                            <span class="entity-tag">${e.text} <span class="entity-label">${e.label}</span></span>
                        `).join('')}
                    </div>
                ` : '<div class="empty-state-mini">No entities extracted</div>'}
            </div>
        </div>
        
        <!-- Row 4: Transcript -->
        <div class="detail-card full-width" style="grid-column: 1/-1;">
            <h3>🎙️ Transcript</h3>
            <div class="transcript-container">
                ${data.turns.map(t => `
                    <div class="turn-line">
                        <span class="turn-speaker ${t.speaker}">${t.speaker}:</span>
                        <span class="turn-text">${t.text}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// ─── RAG Query ───────────────────────────────────────────
function setRAGQuery(query) {
    document.getElementById('rag-query-input').value = query;
    queryRAG();
}

async function queryRAG() {
    const input = document.getElementById('rag-query-input');
    const query = input.value.trim();
    if (!query) return;
    
    const container = document.getElementById('rag-results');
    container.innerHTML = '<div class="shimmer" style="height:200px;border-radius:16px;"></div>';
    
    try {
        const res = await fetch(`${API_BASE}/api/rag/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 5 })
        });
        
        if (!res.ok) throw new Error('Query failed');
        
        const data = await res.json();
        
        let html = '';
        
        // AI Answer
        if (data.answer) {
            html += `
                <div class="rag-answer">
                    <div class="rag-answer-header">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                        AI Answer
                    </div>
                    <p>${data.answer}</p>
                </div>
            `;
        }
        
        // Sources
        if (data.results && data.results.length > 0) {
            html += '<h3 style="margin-bottom: 12px; font-size: 0.88rem; color: var(--text-muted);">Sources</h3>';
            data.results.forEach((r, i) => {
                html += `
                    <div class="rag-source">
                        <div class="rag-source-header">
                            <span>Call: ${r.call_id}</span>
                            <span>Relevance: ${(r.score * 100).toFixed(0)}%</span>
                        </div>
                        <p>${r.text}</p>
                    </div>
                `;
            });
        } else if (!data.answer) {
            html = `
                <div class="empty-state">
                    <h3>No results found</h3>
                    <p>Try a different query or upload more calls to build your knowledge base</p>
                </div>
            `;
        }
        
        container.innerHTML = html;
        
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><h3>Query failed</h3><p>${e.message}</p></div>`;
        showToast('RAG query failed: ' + e.message, 'error');
    }
}

// Enter key for RAG input
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && document.activeElement.id === 'rag-query-input') {
        queryRAG();
    }
});

// ─── Utilities ───────────────────────────────────────────
function getScoreClass(score) {
    if (score >= 80) return 'excellent';
    if (score >= 60) return 'good';
    if (score >= 40) return 'average';
    return 'poor';
}

function getScoreColor(score) {
    if (score >= 80) return 'var(--success)';
    if (score >= 60) return 'var(--info)';
    if (score >= 40) return 'var(--warning)';
    return 'var(--danger)';
}

function getSentimentEmoji(label) {
    if (label === 'POSITIVE') return '😊';
    if (label === 'NEGATIVE') return '😠';
    return '😐';
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
