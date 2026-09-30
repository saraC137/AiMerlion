// ================================================================
// 📊 STATS MODAL — Open / Close / Render
// ================================================================

/**
 * openStats()
 * 
 * Fetches fresh stats from /api/stats and opens the modal.
 * Now with proper error handling and visual feedback! 💅
 * 
 * Think of it as opening the backstage door — you might find 
 * a perfectly organized dressing room, or you might find chaos.
 * Either way, we handle it gracefully! 🎭
 */
function openStats() {
    const overlay = document.getElementById('statsOverlay');
    const statsBtn = document.getElementById('statsBtn');

    // Visual feedback on button — so user knows it's working
    if (statsBtn) {
        statsBtn.textContent = '⏳ Loading...';
        statsBtn.disabled = true;
    }

    // Open the overlay immediately with loading state
    overlay.classList.add('open');
    document.addEventListener('keydown', closeStatsOnEscape);

    // Reset to loading state
    document.getElementById('statsBody').innerHTML = `
        <div class="stats-loading">
            <div class="spin">⚙️</div>
            <div>Crunching the numbers, darling…</div>
        </div>`;
    document.getElementById('statsSubtitle').textContent = 'Loading…';

    // Fetch fresh data every time the modal opens
    fetch('/api/stats')
        .then(r => {
            if (!r.ok) throw new Error(`Server returned ${r.status} ${r.statusText}`);
            return r.json();
        })
        .then(d => {
            renderStats(d);
            // Restore button state
            if (statsBtn) { statsBtn.textContent = '📊 Stats'; statsBtn.disabled = false; }
        })
        .catch(err => {
            console.error('❌ Stats fetch failed:', err);
            document.getElementById('statsBody').innerHTML = `
                <div class="stats-error">
                    ❌ Failed to load stats.<br>
                    <small style="color:var(--text-muted);">${escapeHtml(String(err))}</small>
                    <br><br>
                    <div style="font-size:0.75rem; color:var(--text-muted);">
                        💡 This usually means the database hasn't been set up yet,
                        or the server isn't running. Try saving at least one annotation first!
                    </div>
                </div>`;
            // Restore button state
            if (statsBtn) { statsBtn.textContent = '📊 Stats'; statsBtn.disabled = false; }
        });
}

function closeStats() {
    document.getElementById('statsOverlay').classList.remove('open');
    document.removeEventListener('keydown', closeStatsOnEscape);
    // Restore stats button text in case it was left in loading state
    const statsBtn = document.getElementById('statsBtn');
    if (statsBtn) { statsBtn.textContent = '📊 Stats'; statsBtn.disabled = false; }
}
function closeStatsOnBackdrop(e) {
    if (e.target === document.getElementById('statsOverlay')) closeStats();
}
function closeStatsOnEscape(e) {
    if (e.key === 'Escape') closeStats();
}

/**
 * renderStats(d)
 *
 * Takes the rich stats JSON from /api/stats and builds the full
 * modal content — KPI cards, donut ring, entity bars, leaderboard,
 * activity sparkline, and human/auto ratio bar.
 *
 * Everything is pure HTML/CSS — no charting library required! ✨
 *
 * @param {Object} d - Stats data from /api/stats
 */
function renderStats(d) {
    // ── Derived values ────────────────────────────────────────────────
    const total       = d.queue_total       || 0;
    const completed   = d.queue_completed   || 0;
    const inProgress  = d.queue_in_progress || 0;
    const pending     = d.queue_pending     || 0;
    const totalAnns   = d.total_annotations || 0;
    const humanAnns   = d.human_annotations || 0;
    const autoAnns    = d.auto_annotations  || 0;

    const completePct    = total > 0 ? (completed / total * 100)   : 0;
    const inProgressPct  = total > 0 ? (inProgress / total * 100)  : 0;
    const humanPct       = totalAnns > 0 ? (humanAnns / totalAnns * 100) : 0;

    // Update subtitle
    document.getElementById('statsSubtitle').textContent =
        `${total.toLocaleString()} candidates · ${totalAnns.toLocaleString()} annotations`;

    // ── KPI Cards ─────────────────────────────────────────────────────
    const kpiHTML = `
    <div class="kpi-grid">
        <div class="kpi-card" style="--kpi-color: var(--accent-cyan)">
            <span class="kpi-icon">👥</span>
            <div class="kpi-value">${total.toLocaleString()}</div>
            <div class="kpi-label">Total Candidates</div>
        </div>
        <div class="kpi-card" style="--kpi-color: var(--accent-green)">
            <span class="kpi-icon">✅</span>
            <div class="kpi-value">${completed.toLocaleString()}</div>
            <div class="kpi-label">Completed</div>
            <div class="kpi-sub">${completePct.toFixed(1)}% of queue</div>
        </div>
        <div class="kpi-card" style="--kpi-color: var(--accent-yellow)">
            <span class="kpi-icon">🔵</span>
            <div class="kpi-value">${inProgress.toLocaleString()}</div>
            <div class="kpi-label">In Progress</div>
            <div class="kpi-sub">${inProgressPct.toFixed(1)}% of queue</div>
        </div>
        <div class="kpi-card" style="--kpi-color: var(--accent-purple)">
            <span class="kpi-icon">🏷️</span>
            <div class="kpi-value">${totalAnns.toLocaleString()}</div>
            <div class="kpi-label">Total Annotations</div>
            <div class="kpi-sub">${humanAnns.toLocaleString()} human · ${autoAnns.toLocaleString()} auto</div>
        </div>
    </div>`;

    // ── Donut ring — completion progress ─────────────────────────────
    // SVG donut: circumference = 2π × r  (r = 54)
    const R = 54;
    const CIRC = 2 * Math.PI * R;
    const completeDash  = (completePct  / 100) * CIRC;
    const inProgDash    = (inProgressPct / 100) * CIRC;
    // Completed arc offset = 0 (starts at top after rotate(-90deg))
    // InProgress arc starts after the completed section
    const inProgOffset  = CIRC - completeDash;

    const donutHTML = `
    <div class="donut-wrap">
        <h4>📋 Queue Progress</h4>
        <div class="donut-container">
            <svg class="donut-svg" width="130" height="130" viewBox="0 0 130 130">
                <!-- Track (background ring) -->
                <circle class="donut-track" cx="65" cy="65" r="${R}"/>
                <!-- In-Progress arc (yellow, behind completed) -->
                <circle class="donut-fill in-progress" cx="65" cy="65" r="${R}"
                    stroke-dasharray="${inProgDash} ${CIRC}"
                    stroke-dashoffset="${-completeDash}"/>
                <!-- Completed arc (cyan, on top) -->
                <circle class="donut-fill" cx="65" cy="65" r="${R}"
                    stroke-dasharray="${completeDash} ${CIRC}"
                    stroke-dashoffset="0"/>
            </svg>
            <div class="donut-center">
                <div class="donut-pct">${completePct.toFixed(0)}%</div>
                <div class="donut-pct-label">Done</div>
            </div>
        </div>
        <div class="donut-legend">
            <div class="donut-legend-item">
                <div class="donut-legend-dot" style="background:var(--accent-green)"></div>
                <span class="donut-legend-label">Completed</span>
                <span class="donut-legend-val">${completed.toLocaleString()}</span>
            </div>
            <div class="donut-legend-item">
                <div class="donut-legend-dot" style="background:var(--accent-yellow)"></div>
                <span class="donut-legend-label">In Progress</span>
                <span class="donut-legend-val">${inProgress.toLocaleString()}</span>
            </div>
            <div class="donut-legend-item">
                <div class="donut-legend-dot" style="background:var(--bg-elevated); border:1px solid var(--border)"></div>
                <span class="donut-legend-label">Pending</span>
                <span class="donut-legend-val">${pending.toLocaleString()}</span>
            </div>
        </div>
    </div>`;

    // ── Entity distribution bars ──────────────────────────────────────
    const entityCounts = d.entity_counts || {};
    const entityColors = d.entity_colors || {};
    const maxCount = Math.max(1, ...Object.values(entityCounts));

    // Sort by count descending, take top 15
    const topEntities = Object.entries(entityCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 15);

    const entityBarsHTML = topEntities.map(([name, count]) => {
        const pct   = (count / maxCount * 100).toFixed(1);
        const color = entityColors[name] || '#7ee8fa';
        // Convert ENTITY_NAME → Entity Name for display
        const label = name.replace(/_/g, ' ')
                          .toLowerCase()
                          .replace(/\b\w/g, c => c.toUpperCase());
        return `
        <div class="entity-bar-row">
            <div class="entity-bar-label" title="${name}">${label}</div>
            <div class="entity-bar-track">
                <div class="entity-bar-fill"
                     style="width:${pct}%; background:${color};"
                     title="${count.toLocaleString()} annotations"></div>
            </div>
            <div class="entity-bar-count">${count.toLocaleString()}</div>
        </div>`;
    }).join('');

    const entityChartHTML = `
    <div class="entity-chart-wrap">
        <h4>🏷️ Entity Distribution <span style="color:var(--text-muted); font-weight:400">(top 15 by count)</span></h4>
        ${topEntities.length > 0 ? entityBarsHTML :
          '<div style="color:var(--text-muted); font-size:0.82rem; padding:20px 0; text-align:center;">No annotations yet — get labelling! 💅</div>'}
    </div>`;

    // ── Leaderboard ───────────────────────────────────────────────────
    const medals = ['🥇','🥈','🥉','4️⃣','5️⃣'];
    const leaderHTML = (d.leaderboard || []).length > 0
        ? (d.leaderboard || []).map((item, i) => `
            <div class="leader-row">
                <div class="leader-rank" style="color:${i < 3 ? 'var(--accent-yellow)' : 'var(--text-muted)'}">${medals[i] || (i+1)}</div>
                <div class="leader-name">${escapeHtml(item.annotator)}</div>
                <div class="leader-stat">
                    <strong>${item.annotations.toLocaleString()}</strong>
                    <span> anns · ${item.docs} docs</span>
                </div>
            </div>`).join('')
        : '<div style="color:var(--text-muted); font-size:0.8rem; padding:16px 0; text-align:center;">No annotator data yet.</div>';

    const leaderboardHTML = `
    <div class="stats-panel">
        <h4>🏆 Annotator Leaderboard</h4>
        ${leaderHTML}
    </div>`;

    // ── Activity sparkline ─────────────────────────────────────────────
    // Fill in any missing days in the last 7 days so we always show 7 bars
    const activity = d.recent_activity || [];
    const today = new Date();
    const days = [];
    for (let i = 6; i >= 0; i--) {
        const d2 = new Date(today);
        d2.setDate(today.getDate() - i);
        const key = d2.toISOString().slice(0, 10);
        const label = d2.toLocaleDateString('en-GB', {weekday:'short'});
        const found = activity.find(a => a.day === key);
        days.push({ label, count: found ? found.count : 0 });
    }
    const maxAct = Math.max(1, ...days.map(d => d.count));

    const activityBarsHTML = days.map(day => {
        const barPct = (day.count / maxAct * 100);
        const barHeight = Math.max(2, Math.round(barPct * 0.7)); // max 70px
        return `
        <div class="activity-bar-wrap" title="${day.label}: ${day.count} annotations">
            <div class="activity-bar" style="height:${barHeight}px"></div>
            <div class="activity-day">${day.label}</div>
        </div>`;
    }).join('');

    // ── Human vs Auto ratio ────────────────────────────────────────────
    const ratioHTML = `
    <div style="margin-top:16px;">
        <div style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.6px; font-weight:700; margin-bottom:8px;">
            ✍️ Human vs 🤖 Auto
        </div>
        <div class="ratio-bar-wrap">
            <div class="ratio-bar-track">
                <div class="ratio-bar-human" style="width:${humanPct}%"></div>
                <div class="ratio-bar-auto"></div>
            </div>
            <div class="ratio-labels">
                <div class="ratio-label">
                    <div class="ratio-dot" style="background:var(--accent-green)"></div>
                    Human: <strong style="color:var(--accent-green); margin-left:3px">${humanAnns.toLocaleString()}</strong>
                    &nbsp;<span style="color:var(--text-muted)">(${humanPct.toFixed(1)}%)</span>
                </div>
                <div class="ratio-label">
                    <div class="ratio-dot" style="background:var(--accent-purple)"></div>
                    Auto: <strong style="color:var(--accent-purple); margin-left:3px">${autoAnns.toLocaleString()}</strong>
                    &nbsp;<span style="color:var(--text-muted)">(${(100 - humanPct).toFixed(1)}%)</span>
                </div>
            </div>
        </div>
    </div>`;

    const activityHTML = `
    <div class="stats-panel">
        <h4>📅 Last 7 Days Activity</h4>
        <div class="activity-chart">${activityBarsHTML}</div>
        ${ratioHTML}
    </div>`;

    // ── Assemble everything into the modal body ────────────────────────
    document.getElementById('statsBody').innerHTML = `
        ${kpiHTML}
        <div class="stats-charts-row">
            ${donutHTML}
            ${entityChartHTML}
        </div>
        <div class="stats-bottom-row">
            ${leaderboardHTML}
            ${activityHTML}
        </div>`;
}

// Reuse escapeHtml if available (defined in annotate template),
// or define a local fallback here for the index page
function escapeHtml(text) {
    if (text == null) return '';
    const str = String(text);
    const div = document.createElement('div');
    div.textContent = str;
    // innerHTML escapes <, >, & — but NOT quotes!
    // We add quote escaping for safe attribute injection.
    return div.innerHTML
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/**
 * decodeHtmlEntities(text)
 *
 * 🐛 BUG FIX HELPER: Decodes HTML entities back to their original characters.
 * Used by textarea oninput handlers to compare current value against the
 * original (which was HTML-escaped for safe storage in data-original).
 *
 * @param {string} text - HTML-encoded text
 * @returns {string}    - Decoded original text
 */
function decodeHtmlEntities(text) {
    if (text == null) return '';
    const div = document.createElement('div');
    div.innerHTML = text;
    return div.textContent || div.innerText || '';
}

