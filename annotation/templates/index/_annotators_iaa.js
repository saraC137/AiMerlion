// ================================================================
// 🍞 TOAST — Index page
// ================================================================
function showToast(msg, type = 'success', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.style.cssText = `
        padding:12px 20px; border-radius:8px; margin-bottom:8px; font-size:0.82rem;
        color:#e6edf3; max-width:480px; box-shadow:0 4px 24px rgba(0,0,0,0.4);
        background:${type === 'error' ? 'rgba(218,54,51,0.9)' : 'rgba(86,211,100,0.9)'};
    `;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ================================================================
// 👥 ANNOTATOR MANAGER — Add / Select / Delete
// 💅 Fairy Codemother's Glow-Up Edition v2.0! ✨
// ================================================================

/** Track which card is in "confirm delete" mode */
let amDeletePending = null;

function openAnnotatorManager() {
    document.getElementById('annotatorManagerOverlay').classList.add('open');
    document.addEventListener('keydown', closeAnnotatorManagerOnEscape);
    amDeletePending = null;
    refreshAnnotatorManagerList();
    // Auto-focus the add input after a beat
    setTimeout(() => document.getElementById('newAnnotatorInput').focus(), 200);
}

function closeAnnotatorManager() {
    document.getElementById('annotatorManagerOverlay').classList.remove('open');
    document.removeEventListener('keydown', closeAnnotatorManagerOnEscape);
    amDeletePending = null;
}

function closeAnnotatorManagerOnEscape(e) {
    if (e.key === 'Escape') closeAnnotatorManager();
}

function closeAnnotatorManagerOnBackdrop(e) {
    if (e.target === document.getElementById('annotatorManagerOverlay')) closeAnnotatorManager();
}

function refreshAnnotatorManagerList() {
    const container = document.getElementById('annotatorListContainer');
    const currentA = localStorage.getItem('annotator_name') || '';
    const currentB = localStorage.getItem('iaa_annotator_name') || '';

    // ── Update status cards ──────────────────────────────────────
    document.getElementById('currentALabel').textContent = currentA || '(click Set A below)';
    document.getElementById('currentBLabel').textContent = currentB || '(click Set B below)';
    if (!currentA) document.getElementById('currentALabel').style.opacity = '0.5';
    else document.getElementById('currentALabel').style.opacity = '1';
    if (!currentB) document.getElementById('currentBLabel').style.opacity = '0.5';
    else document.getElementById('currentBLabel').style.opacity = '1';

    fetch('/api/annotators')
        .then(r => r.json())
        .then(data => {
            const list = data.annotators || [];
            document.getElementById('amCount').textContent = list.length;

            if (list.length === 0) {
                container.innerHTML = `
                    <div class="am-empty">
                        <div style="font-size:2rem; margin-bottom:8px;">🎭</div>
                        <div style="font-size:0.85rem; margin-bottom:4px;">No annotators yet!</div>
                        <div style="font-size:0.72rem;">Type a name above and click <strong>+ Add</strong> to begin.</div>
                    </div>`;
                return;
            }

            let html = '';
            list.forEach(a => {
                const isA = currentA && currentA.toLowerCase() === a.name.toLowerCase();
                const isB = currentB && currentB.toLowerCase() === a.name.toLowerCase();
                const isAB = isA && isB;  // Edge: shouldn't happen but handle gracefully

                // ── Avatar class + initials ──────────────────────
                const initials = a.name.split(/\s+/).map(w => w[0]).join('').toUpperCase().slice(0, 2);
                const avatarClass = isAB ? 'am-avatar-ab' : isA ? 'am-avatar-a' : isB ? 'am-avatar-b' : 'am-avatar-none';

                // ── Card border class ────────────────────────────
                const cardClass = isAB ? 'am-active-ab' : isA ? 'am-active-a' : isB ? 'am-active-b' : '';

                // ── Stats line ───────────────────────────────────
                const stats = [];
                if (a.primary_docs > 0) stats.push(`${a.primary_docs} doc${a.primary_docs !== 1 ? 's' : ''} annotated`);
                if (a.iaa_docs > 0) stats.push(`${a.iaa_docs} IAA`);
                const statsText = stats.length > 0 ? stats.join(' · ') : 'No activity yet';

                // ── Delete area (inline confirm) ─────────────────
                const isDeleting = amDeletePending === a.id;
                const deleteHTML = isDeleting
                    ? `<div class="am-confirm">
                         <span style="font-size:0.63rem;color:var(--accent-red);">Delete?</span>
                         <button class="am-confirm-btn am-confirm-yes"
                                 onclick="event.stopPropagation();confirmDeleteAnnotator(${a.id},'${escapeHtml(a.name)}')">Yes</button>
                         <button class="am-confirm-btn am-confirm-no"
                                 onclick="event.stopPropagation();cancelDeleteAnnotator()">No</button>
                       </div>`
                    : `<button class="am-del"
                               onclick="event.stopPropagation();startDeleteAnnotator(${a.id})"
                               title="Remove ${escapeHtml(a.name)}">🗑</button>`;

                html += `
                <div class="am-card ${cardClass}">
                    <div class="am-avatar ${avatarClass}">${initials}</div>
                    <div style="flex:1;min-width:0;">
                        <div class="am-name">${escapeHtml(a.name)}</div>
                        <div class="am-meta">${statsText}</div>
                    </div>
                    <div class="am-pills">
                        <button class="am-pill ${isA ? 'am-pill-a-on' : ''}"
                                onclick="selectAnnotatorAs('A','${escapeHtml(a.name)}')"
                                title="Set as Annotator A (primary)">
                            ${isA ? '✓ A' : 'Set A'}
                        </button>
                        <button class="am-pill ${isB ? 'am-pill-b-on' : ''}"
                                onclick="selectAnnotatorAs('B','${escapeHtml(a.name)}')"
                                title="Set as Annotator B (IAA)">
                            ${isB ? '✓ B' : 'Set B'}
                        </button>
                    </div>
                    ${deleteHTML}
                </div>`;
            });

            container.innerHTML = html;
        })
        .catch(err => {
            container.innerHTML = `<div class="am-empty" style="color:var(--accent-red);">
                Failed to load: ${err}</div>`;
        });
}

function addAnnotatorFromModal() {
    const input = document.getElementById('newAnnotatorInput');
    const name = input.value.trim();

    if (!name || name.length < 2) {
        showToast('Name must be at least 2 characters!', 'error', 3000);
        input.focus();
        return;
    }

    // ── Disable button during request to prevent double-add ──────
    const btn = input.nextElementSibling?.nextElementSibling || null;

    fetch('/api/annotators', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, role: 'both' })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            input.value = '';
            showToast(`👥 "${name}" added! ✨`, 'success', 3000);
            refreshAnnotatorManagerList();
            input.focus();
        } else {
            showToast('❌ ' + (data.error || 'Failed to add'), 'error', 4000);
            input.focus();
        }
    })
    .catch(err => showToast('❌ Network error: ' + err, 'error', 4000));
}

function selectAnnotatorAs(role, name) {
    const currentA = localStorage.getItem('annotator_name') || '';
    const currentB = localStorage.getItem('iaa_annotator_name') || '';

    if (role === 'A') {
        if (currentB && name.toLowerCase() === currentB.toLowerCase()) {
            showToast(`❌ "${name}" is already set as B! A and B must be different! 👯`, 'error', 4000);
            return;
        }
        localStorage.setItem('annotator_name', name);
        showToast(`👤 Annotator A → "${name}" ✨`, 'success', 2500);
    } else {
        if (currentA && name.toLowerCase() === currentA.toLowerCase()) {
            showToast(`❌ "${name}" is already set as A! A and B must be different! 👯`, 'error', 4000);
            return;
        }
        localStorage.setItem('iaa_annotator_name', name);
        showToast(`📏 Annotator B → "${name}" ✨`, 'success', 2500);
    }

    refreshAnnotatorManagerList();
}

/** Step 1: Show inline confirm on the card */
function startDeleteAnnotator(id) {
    amDeletePending = id;
    refreshAnnotatorManagerList();
}

/** Cancel: hide inline confirm */
function cancelDeleteAnnotator() {
    amDeletePending = null;
    refreshAnnotatorManagerList();
}

/** Step 2: Actually delete after confirm */
function confirmDeleteAnnotator(id, name) {
    const currentA = localStorage.getItem('annotator_name') || '';
    const currentB = localStorage.getItem('iaa_annotator_name') || '';

    fetch(`/api/annotators/${id}`, { method: 'DELETE' })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            // Clear from localStorage if they were the active annotator
            if (currentA.toLowerCase() === name.toLowerCase()) {
                localStorage.removeItem('annotator_name');
            }
            if (currentB.toLowerCase() === name.toLowerCase()) {
                localStorage.removeItem('iaa_annotator_name');
            }
            amDeletePending = null;
            showToast(`🗑 "${name}" removed`, 'success', 3000);
            refreshAnnotatorManagerList();
        } else {
            showToast('❌ ' + (data.error || 'Delete failed'), 'error', 4000);
        }
    })
    .catch(err => showToast('❌ Network error: ' + err, 'error', 4000));
}

// ================================================================
// 📏 IAA GLOBAL TOGGLE — Queue page on/off switch
// ================================================================
// ================================================================
// 👥 ANNOTATOR TRACKER — Open / Close / Render
// ================================================================
function openAnnotatorTracker() {
    const overlay = document.getElementById('trackerOverlay');
    overlay.classList.add('open');
    document.addEventListener('keydown', closeTrackerOnEscape);
    document.getElementById('trackerBody').innerHTML = `
        <div class="stats-loading"><div class="spin">⚙️</div><div>Loading annotator data…</div></div>`;
    document.getElementById('trackerSubtitle').textContent = 'Loading…';
    fetch('/api/annotator-tracker')
        .then(r => r.json())
        .then(d => renderAnnotatorTracker(d))
        .catch(err => {
            document.getElementById('trackerBody').innerHTML =
                `<div style="text-align:center;padding:40px;color:#da3633;">Failed to load: ${err}</div>`;
        });
}
function closeTracker() {
    document.getElementById('trackerOverlay').classList.remove('open');
    document.removeEventListener('keydown', closeTrackerOnEscape);
}
function closeTrackerOnEscape(e) { if (e.key === 'Escape') closeTracker(); }
function closeTrackerOnBackdrop(e) {
    if (e.target === document.getElementById('trackerOverlay')) closeTracker();
}

function renderAnnotatorTracker(d) {
    const body = document.getElementById('trackerBody');
    document.getElementById('trackerSubtitle').textContent =
        `${d.total_annotators || 0} annotators · ${d.total_docs || 0} documents tracked`;

    // ── Annotator Summary Cards ──────────────────────────────
    let annoHTML = '';
    if ((d.annotators || []).length > 0) {
        const medals = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣','7️⃣','8️⃣'];
        annoHTML = `<div class="stats-panel"><h4>👥 Annotator Summary</h4>
            ${d.annotators.map((a, i) => {
                const pct = a.total_docs > 0 ? Math.round(a.completed / a.total_docs * 100) : 0;
                const barColor = pct >= 80 ? '#56d364' : pct >= 50 ? '#e3b341' : '#f0883e';
                return `<div style="padding:10px 0; border-bottom:1px solid rgba(42,49,64,0.5);">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                        <span style="font-size:1.1rem;">${medals[i] || ''}</span>
                        <strong style="color:#e6edf3;">${escapeHtml(a.annotator)}</strong>
                        <span style="font-size:0.65rem; color:#545d68; margin-left:auto;">
                            Last: ${a.last_activity ? a.last_activity.slice(0,16).replace('T',' ') : '—'}</span>
                    </div>
                    <div style="display:flex; gap:16px; font-size:0.72rem; color:#8b949e; margin-bottom:6px;">
                        <span>✅ ${a.completed||0} done</span>
                        <span>🔵 ${a.in_progress||0} wip</span>
                        <span>🏷️ ${(a.total_spans||0).toLocaleString()} spans</span>
                        <span>📄 ${a.total_docs||0} docs</span>
                    </div>
                    <div style="background:rgba(42,49,64,0.5); border-radius:4px; height:6px; overflow:hidden;">
                        <div style="background:${barColor}; height:100%; width:${pct}%; border-radius:4px;"></div>
                    </div>
                    <div style="font-size:0.6rem; color:#545d68; margin-top:2px; text-align:right;">${pct}% complete</div>
                </div>`;
            }).join('')}
        </div>`;
    } else {
        annoHTML = `<div class="stats-panel"><h4>👥 Annotator Summary</h4>
            <div style="color:#545d68; font-size:0.82rem; padding:20px 0; text-align:center;">
                No annotators recorded yet. Names are saved when you click 💾 Save or ✅ Complete.
            </div></div>`;
    }

    // ── IAA Sessions ─────────────────────────────────────────
    let iaaHTML = '';
    if ((d.iaa_sessions || []).length > 0) {
        iaaHTML = `<div class="stats-panel" style="margin-top:14px;"><h4>📏 IAA Sessions</h4>
            <div style="max-height:200px; overflow-y:auto;">
            <table style="width:100%; border-collapse:collapse; font-size:0.72rem;">
                <thead><tr style="color:#545d68; text-align:left;">
                    <th style="padding:6px 8px;">Doc</th><th style="padding:6px 8px;">Annotator A</th>
                    <th style="padding:6px 8px;">Annotator B</th><th style="padding:6px 8px;">Spans</th>
                    <th style="padding:6px 8px;">Date</th>
                </tr></thead><tbody>
                ${d.iaa_sessions.map(s => `<tr style="border-bottom:1px solid rgba(42,49,64,0.3);">
                    <td style="padding:6px 8px;"><a href="/annotate/${s.candidate_id||0}" style="color:#7ee8fa;">${escapeHtml(s.doc_id||'—')}</a></td>
                    <td style="padding:6px 8px; color:#8b949e;">${escapeHtml(s.annotator_a||'—')}</td>
                    <td style="padding:6px 8px; color:#f778ba; font-weight:600;">${escapeHtml(s.annotator_b||'—')}</td>
                    <td style="padding:6px 8px;">${s.span_count||0}</td>
                    <td style="padding:6px 8px; color:#545d68;">${s.last_activity ? s.last_activity.slice(0,10) : '—'}</td>
                </tr>`).join('')}
                </tbody></table></div></div>`;
    }

    // ── Recent Docs ──────────────────────────────────────────
    let docsHTML = '';
    if ((d.documents || []).length > 0) {
        docsHTML = `<div class="stats-panel" style="margin-top:14px;"><h4>📋 Recent Documents</h4>
            <div style="max-height:300px; overflow-y:auto;">
            <table style="width:100%; border-collapse:collapse; font-size:0.72rem;">
                <thead><tr style="color:#545d68; text-align:left;">
                    <th style="padding:6px 8px;">ID</th><th style="padding:6px 8px;">Candidate</th>
                    <th style="padding:6px 8px;">Annotator</th><th style="padding:6px 8px;">Status</th>
                    <th style="padding:6px 8px;">Spans</th><th style="padding:6px 8px;">Last Saved</th>
                </tr></thead><tbody>
                ${d.documents.map(doc => {
                    const sc = doc.status==='completed' ? '#56d364' : doc.status==='in_progress' ? '#7ee8fa' : '#e3b341';
                    return `<tr style="border-bottom:1px solid rgba(42,49,64,0.3);">
                        <td style="padding:6px 8px;"><a href="/annotate/${doc.candidate_id||0}" style="color:#7ee8fa;">#${doc.candidate_id||'?'}</a></td>
                        <td style="padding:6px 8px; color:#8b949e; max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(doc.candidate_name||'—')}</td>
                        <td style="padding:6px 8px; color:#eeb8ff; font-weight:600;">${escapeHtml(doc.annotator||'—')}</td>
                        <td style="padding:6px 8px;"><span style="color:${sc};">${doc.status||'pending'}</span></td>
                        <td style="padding:6px 8px;">${doc.span_count||0}</td>
                        <td style="padding:6px 8px; color:#545d68;">${doc.updated_at ? doc.updated_at.slice(0,16).replace('T',' ') : '—'}</td>
                    </tr>`;
                }).join('')}
                </tbody></table></div></div>`;
    }

    body.innerHTML = `${annoHTML}${iaaHTML}${docsHTML}`;
}

function toggleIAAGlobal() {
    const isActive = localStorage.getItem('iaa_mode_active') === 'true';
    if (isActive) {
        localStorage.removeItem('iaa_mode_active');
        updateIAAQueueUI(false);
        showToast('📏 IAA Mode deactivated. Normal annotation restored.', 'success', 4000);
    } else {
        // ── Check if Annotator B was already set via Annotator Manager ──
        let name = localStorage.getItem('iaa_annotator_name') || '';
        const primaryName = localStorage.getItem('annotator_name') || '';

        // Only prompt if no Annotator B has been chosen yet
        if (!name) {
            showToast(
                '📏 Set Annotator B first! Open 👥 Annotators and click "Set B" on a name.',
                'error', 5000
            );
            // Auto-open the manager so they can pick
            openAnnotatorManager();
            return;
        }

        // Guard: A and B must be different
        if (primaryName && name.toLowerCase() === primaryName.toLowerCase()) {
            showToast(
                `❌ Annotator A and B are both "${name}"! Open 👥 Annotators and set a DIFFERENT person as B.`,
                'error', 5000
            );
            openAnnotatorManager();
            return;
        }

        localStorage.setItem('iaa_mode_active', 'true');
        updateIAAQueueUI(true);
        showToast(`📏 IAA ON! Annotating as "${name}" (Annotator B). Open any candidate for a clean slate!`, 'success', 5000);
    }
}

function updateIAAQueueUI(active) {
    const btn = document.getElementById('iaaToggleBtn');
    const banner = document.getElementById('iaaQueueBanner');
    const nameEl = document.getElementById('iaaQueueName');

    // ── Show/hide the IAA Status column in the table ─────────────
    document.querySelectorAll('.iaa-col').forEach(el => {
        el.style.display = active ? '' : 'none';
    });

    if (active) {
        const name = localStorage.getItem('iaa_annotator_name') || '?';
        if (btn) { btn.textContent = '📏 IAA ON'; btn.style.background = 'rgba(247,120,186,0.2)'; btn.style.fontWeight = '700'; }
        if (banner) banner.style.display = 'flex';
        if (nameEl) nameEl.textContent = name;
    } else {
        if (btn) { btn.textContent = '📏 IAA Mode'; btn.style.background = ''; btn.style.fontWeight = ''; }
        if (banner) banner.style.display = 'none';
    }
}

// Restore IAA state on page load
(function() {
    if (localStorage.getItem('iaa_mode_active') === 'true') updateIAAQueueUI(true);
})();

