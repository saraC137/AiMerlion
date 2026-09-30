// =================================================================
// 🤖 PRE-ANNOTATE
// =================================================================
function runPreAnnotate() {
    const threshold = parseFloat(prompt('Confidence threshold (0.0 - 1.0):', '0.6'));
    if (isNaN(threshold)) return;

    fetch(`/api/pre-annotate/${CANDIDATE_ID}?threshold=${threshold}`)
    .then(r => r.json())
    .then(d => {
        if (d.annotations) {
            // Merge: keep human annotations, add new auto ones that don't overlap
            const humanAnns = annotations.filter(a => a.annotator === 'human');
            const autoAnns = d.annotations.filter(newA => {
                return !humanAnns.some(h =>
                    !(newA.char_end <= h.char_start || newA.char_start >= h.char_end)
                );
            });
            annotations = [...humanAnns, ...autoAnns];
            renderText();
            showToast(`🤖 Added ${autoAnns.length} auto-annotations (kept ${humanAnns.length} human)`);
        }
    });
}

// =================================================================
// 🏷️ BIO PREVIEW — Now lives in its own glorious tab! ✨
// =================================================================

/**
 * loadBIOPreview()
 * 
 * Fetches BIO tags from the server and renders them in the BIO tab.
 * Previously this was a toggle — now it's a proper tab citizen.
 * Like upgrading from a flip phone to a smartphone, darling! 📱✨
 */
function loadBIOPreview() {
    const preview = document.getElementById('bioPreview');
    const statsEl = document.getElementById('bioStats');

    // Show loading state
    preview.innerHTML = '<div style="color:var(--text-muted); font-size:0.78rem; text-align:center; padding:20px 0;">⏳ Loading BIO tags...</div>';

    fetch(`/api/bio-preview/${CANDIDATE_ID}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({annotations})  // Send current annotations state
    })
    .then(r => r.json())
    .then(d => {
        const total = d.total_tokens;
        const oCount = d.tag_counts['O'] || 0;
        const labeled = total - oCount;
        const coverage = total > 0 ? ((labeled / total) * 100).toFixed(1) : 0;

        // Update the stats line above the preview box
        if (statsEl) {
            statsEl.innerHTML = `
                <strong style="color:var(--accent-cyan)">${total}</strong> tokens | 
                <strong style="color:var(--accent-green)">${labeled}</strong> labeled 
                (<strong style="color:var(--accent-yellow)">${coverage}%</strong> coverage) | 
                <span style="color:var(--text-muted)">Showing first 200</span>
            `;
        }

        // Render each token-tag pair as a row
        preview.innerHTML = d.preview.map(p => {
            // Color code: O = muted, B- = green (start), I- = cyan (inside)
            const tagClass = p.tag === 'O' ? 'tag-o' :
                             p.tag.startsWith('B-') ? 'tag-b' : 'tag-i';
            return `<div class="bio-row">
                <span class="bio-token">${escapeHtml(p.token)}</span>
                <span class="bio-tag ${tagClass}">${p.tag}</span>
            </div>`;
        }).join('');
    })
    .catch(err => {
        // Graceful error state — no crashes allowed on this runway! 🛑
        preview.innerHTML = `<div style="color:var(--accent-red); font-size:0.78rem; padding:10px 0;">
            ❌ Failed to load BIO preview. Check console for details.
        </div>`;
        console.error('BIO preview error:', err);
    });
}

// Keep old function name as alias so any existing calls still work
function showBIOPreview() {
    openBIOModal();
}

// =================================================================
// 🧩 EDGE CASE CHECK — Now in its own tab! 
// =================================================================

/**
 * loadEdgeCases()
 * 
 * Scans the raw text for potential annotation edge cases and 
 * renders hints in the Edge Cases tab. 
 * Think of edge cases as the divas of the annotation world —
 * they need SPECIAL attention! 💅
 */
function loadEdgeCases() {
    const container = document.getElementById('edgeHints');

    // Show loading state
    container.innerHTML = '<div style="color:var(--text-muted); font-size:0.78rem; text-align:center; padding:20px 0;">⏳ Scanning for edge cases...</div>';

    fetch('/api/edge-case-check', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: RAW_TEXT})
    })
    .then(r => r.json())
    .then(d => {
        if (!d.suggestions?.length) {
            // All clear — no drama today! 🎉
            container.innerHTML = '<div style="color:var(--accent-green); font-size:0.78rem; text-align:center; padding:20px 0;">✅ No edge cases detected! Clean and fabulous! ✨</div>';
            return;
        }
        // Render each edge case hint as a styled card
        container.innerHTML = d.suggestions.map(s =>
            `<div class="edge-hint">
                <strong>${s.type}:</strong> "${escapeHtml(s.text)}"<br>
                💡 ${s.advice}
            </div>`
        ).join('');
    })
    .catch(err => {
        container.innerHTML = `<div style="color:var(--accent-red); font-size:0.78rem;">❌ Scan failed. Check console.</div>`;
        console.error('Edge case check error:', err);
    });
}

// Keep old name as alias for backward compat
function checkEdgeCases() {
    openEdgeModal();
}

