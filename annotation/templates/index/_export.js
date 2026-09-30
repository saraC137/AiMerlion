// ================================================================
// 📦 EXPORT OPTIONS MODAL — Replaces prompt() calls
// ================================================================

/** Tracks what kind of export the user is performing */
let exportMode = null;  // 'csv', 'json', or 'training'

/**
 * openExportModal(mode)
 * 
 * Opens the export options modal with the right fields for each mode.
 * No more prompt() — this is a PROPER UI, darling! 💅✨
 * 
 * prompt() is like showing up to a gala in sweatpants —
 * it WORKS, but the bouncers (browsers) might not let you in! 🚫
 * 
 * @param {string} mode - 'csv', 'json', or 'training'
 */
function openExportModal(mode) {
    exportMode = mode;
    const overlay = document.getElementById('exportOverlay');
    const title = document.getElementById('exportModalTitle');
    const body = document.getElementById('exportModalBody');
    const confirmBtn = document.getElementById('exportConfirmBtn');

    // Reset confirm button
    confirmBtn.disabled = false;
    confirmBtn.textContent = '📤 Export';

    if (mode === 'csv' || mode === 'json') {
        // Profile exports — need status filter
        title.textContent = mode === 'csv' ? '📊 Export CSV Profiles' : '📋 Export JSON Profiles';
        body.innerHTML = `
            <div class="export-field">
                <label>Which records to export?</label>
                <select id="exportStatusSelect">
                    <option value="completed" selected>✅ Completed — fully annotated only</option>
                    <option value="in_progress">🔵 In Progress — include drafts too</option>
                    <option value="all">📋 All — everything in the queue</option>
                </select>
            </div>
            <div class="export-hint">
                💡 <strong>${mode.toUpperCase()}</strong> format — 
                ${mode === 'csv' 
                    ? 'Excel-compatible spreadsheet. Open with Data → Text to Columns for nested fields.'
                    : 'Structured JSON array. Ready for CRM/ATS import.'}
            </div>
        `;
    } else if (mode === 'training') {
        // NER training data export — need format picker
        title.textContent = '🤖 Export Training Data';
        body.innerHTML = `
            <div class="export-field">
                <label>Export format</label>
                <select id="exportFormatSelect" onchange="updateExportHint()">
                    <optgroup label="🏷️ NER Training (token-level)">
                        <option value="conll" selected>📄 CoNLL-2003 — most compatible</option>
                        <option value="spacy">🐍 spaCy v3 JSON</option>
                        <option value="huggingface">🤗 HuggingFace JSONL</option>
                        <option value="custom">📋 Full metadata JSON</option>
                    </optgroup>
                    <optgroup label="🧠 Classification Training (document-level)">
                        <option value="classification">🧠 Classification JSONL — Function &amp; Industry</option>
                    </optgroup>
                </select>
            </div>
            <div class="export-field">
                <label>Status filter</label>
                <select id="exportTrainStatusSelect">
                    <option value="completed" selected>✅ Completed only (recommended)</option>
                    <option value="in_progress">🔵 Include in-progress</option>
                    <option value="all">📋 All documents</option>
                </select>
            </div>
            <div class="export-hint" id="exportHintText">
                💡 NER model training formats — teaches the model to identify entities word-by-word.
            </div>
        `;
    }

    overlay.classList.add('open');

    // ── Auto-scroll the confirm button into view ────────────────────────
    // After a short paint delay, smoothly scroll the confirm button into
    // the visible area so the user never has to hunt for it. 💅
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            const btn = document.getElementById('exportConfirmBtn');
            if (btn) {
                btn.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        });
    });

    // Keyboard: Escape to close, Enter to confirm
    document.addEventListener('keydown', exportKeyHandler);
}

/**
 * confirmExport()
 * 
 * Reads the selected options from the modal and triggers the actual export.
 * The grand finale — lights, camera, DOWNLOAD! 🎬💾
 */
function confirmExport() {
    const confirmBtn = document.getElementById('exportConfirmBtn');
    confirmBtn.disabled = true;
    confirmBtn.textContent = '⏳ Exporting...';

    if (exportMode === 'csv' || exportMode === 'json') {
        // ── Profile export (CSV or JSON) ─────────────────────────────
        const status = document.getElementById('exportStatusSelect').value;

        fetch('/api/export-profiles', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ format: exportMode, status: status })
        })
        .then(r => r.json())
        .then(d => {
            closeExportModal();
            if (d.error) {
                showToast(`❌ Export failed: ${d.error}`, 'error');
                return;
            }
            // Trigger browser file download via hidden <a> tag
            const a = document.createElement('a');
            a.href = d.download_url;
            a.download = d.download_url.split('/').pop();
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            showToast(
                `✅ <strong>${d.documents} profile${d.documents !== 1 ? 's' : ''} exported!</strong> Your ${d.format.toUpperCase()} is downloading — go slay that talent pipeline! 💼🎉`,
                'success',
                5000
            );
        })
        .catch(err => {
            closeExportModal();
            showToast(`❌ Network error: ${err}`, 'error');
        });

    } else if (exportMode === 'training') {
        // ── NER training data export ─────────────────────────────────
        const fmt = document.getElementById('exportFormatSelect').value;
        const status = document.getElementById('exportTrainStatusSelect').value;

        fetch('/api/export', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ format: fmt, status: status })
        })
        .then(r => r.json())
        .then(d => {
            closeExportModal();
            if (d.error) {
                showToast(`❌ ${d.error}`, 'error');
            } else {
                showToast(`🤖 Exported ${d.documents} docs → ${d.path} (${d.format})`, 'success');
            }
        })
        .catch(err => {
            closeExportModal();
            showToast(`❌ Network error: ${err}`, 'error');
        });
    }
}

/**
 * closeExportModal()
 * Hides the export modal and cleans up.
 */
function closeExportModal() {
    document.getElementById('exportOverlay').classList.remove('open');
    document.removeEventListener('keydown', exportKeyHandler);
    exportMode = null;
    // Reset confirm button state
    const btn = document.getElementById('exportConfirmBtn');
    if (btn) { btn.disabled = false; btn.textContent = '📤 Export'; }
}

/**
 * exportKeyHandler(event)
 * Escape → close, Enter → confirm
 */
function exportKeyHandler(event) {
    if (event.key === 'Escape') {
        event.preventDefault();
        closeExportModal();
    }
    if (event.key === 'Enter') {
        event.preventDefault();
        confirmExport();
    }
}

/**
 * exportProfiles(format)
 *
 * Wrapper called by toolbar buttons — now opens the modal
 * instead of using the cursed prompt()! 💅
 *
 * @param {string} format - "json" or "csv"
 */
function exportProfiles(format) {
    openExportModal(format);
}

/**
 * exportTrainingData()
 *
 * Wrapper called by toolbar button — opens modal for NER training export.
 */
function exportTrainingData() {
    openExportModal('training');
}

/**
 * updateExportHint()
 *
 * Dynamically updates the hint text in the export modal based on
 * the selected format. Because NER and Classification are VERY
 * different beasts, and we want the user to know which is which! 🧠🏷️
 */
function updateExportHint() {
    const fmt = document.getElementById('exportFormatSelect').value;
    const hint = document.getElementById('exportHintText');
    if (!hint) return;

    if (fmt === 'classification') {
        hint.innerHTML = '🧠 <strong>Classification training</strong> — exports whole resume text + Function/Industry labels. ' +
            'Teaches the model to predict which <em>department</em> and <em>industry</em> a candidate belongs to. ' +
            'Requires Function &amp; Industry set in the Classify tab.';
    } else {
        hint.innerHTML = '💡 <strong>NER training formats</strong> — exports token-level BIO tags. ' +
            'Teaches the model to identify entities (Name, Phone, Skills, etc.) word-by-word.';
    }
}


