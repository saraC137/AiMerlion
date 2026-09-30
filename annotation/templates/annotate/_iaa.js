// =================================================================
// 📏 IAA — INTER-ANNOTATOR AGREEMENT SYSTEM
// =================================================================
//
// This section handles:
//   1. Annotator name management (localStorage-backed)
//   2. IAA mode toggle (annotator B can label the same doc)
//   3. IAA dashboard with heatmap + metrics
//
// Architecture analogy: Think of IAA like having TWO fashion critics
// review the same runway show independently, then comparing their
// scorecards to see how often they agree. The heatmap shows which
// "categories" (entity types) cause the most disagreement! 👗📊
// =================================================================

/** @type {boolean} Whether we're currently in IAA annotation mode */
let iaaMode = false;

/** @type {boolean} Tracks whether IAA annotations have unsaved changes */
let iaaHasUnsavedChanges = false;

/** @type {string} Scope for the IAA dashboard: 'doc' or 'all' */
let iaaDashboardScope = 'all';

/**
 * @type {string|null} Currently selected Annotator B filter for the dashboard.
 * null = show all annotators aggregated; string = filter to that name only.
 */
let iaaSelectedAnnotator = null;

/**
 * @type {Array} Cached list of annotators from /api/iaa/annotators.
 * Populated once when the dashboard opens, refreshed on demand.
 */
let iaaAnnotatorList = [];

/**
 * setIAASaveStatus(state, spanCount?)
 *
 * Updates the save status indicator in the IAA banner.
 * Like a little scoreboard that always tells Annotator B
 * whether their work is safe! 📋✨
 *
 * States:
 *   'loading'  → ⏳ fetching previous annotations on mode enter
 *   'saved'    → ✅ N spans saved  (shown after successful save or on restore)
 *   'unsaved'  → ● Unsaved changes  (shown when annotations are modified)
 *   'saving'   → 💾 Saving...  (shown while the fetch is in-flight)
 *   'error'    → ⚠️ Save failed  (shown if save request errors out)
 */
function setIAASaveStatus(state, spanCount) {
    const el = document.getElementById('iaaSaveStatus');
    const btn = document.getElementById('iaaSaveBtn');
    if (!el) return;

    switch (state) {
        case 'loading':
            el.textContent = '⏳ Loading previous work...';
            el.style.color = 'var(--text-muted)';
            iaaHasUnsavedChanges = false;
            break;
        case 'saved':
            el.textContent = `✅ ${spanCount != null ? spanCount + ' span' + (spanCount !== 1 ? 's' : '') + ' ' : ''}saved`;
            el.style.color = 'var(--accent-green)';
            iaaHasUnsavedChanges = false;
            // Flash the save button green briefly as confirmation
            if (btn) {
                btn.style.borderColor = 'var(--accent-green)';
                btn.style.color = 'var(--accent-green)';
                setTimeout(() => {
                    btn.style.borderColor = 'var(--accent-pink)';
                    btn.style.color = 'var(--accent-pink)';
                }, 2500);
            }
            break;
        case 'unsaved':
            el.textContent = '● Unsaved changes';
            el.style.color = 'var(--accent-yellow)';
            iaaHasUnsavedChanges = true;
            break;
        case 'saving':
            el.textContent = '💾 Saving...';
            el.style.color = 'var(--text-muted)';
            break;
        case 'error':
            el.textContent = '⚠️ Save failed — try again';
            el.style.color = 'var(--accent-red)';
            iaaHasUnsavedChanges = true;
            break;
        default:
            el.textContent = '';
    }
}

// -----------------------------------------------------------------
// 👤 ANNOTATOR NAME — localStorage-backed identity
// -----------------------------------------------------------------

/**
 * getAnnotatorName()
 *
 * Returns the stored annotator name from localStorage, or empty string.
 * Used to populate the annotator field on saves and IAA sessions.
 *
 * @returns {string} The annotator's name
 */
function getAnnotatorName() {
    return localStorage.getItem('annotator_name') || '';
}

/**
 * setAnnotatorName(name)
 *
 * Stores the annotator name in localStorage for persistence across sessions.
 *
 * @param {string} name - The annotator's display name
 */
function setAnnotatorName(name) {
    localStorage.setItem('annotator_name', name.trim());
}

/**
 * initAnnotatorName()
 *
 * Called on page load. If no annotator name is stored, shows the
 * welcome prompt. Otherwise, silently proceeds.
 *
 * Like checking IDs at the door — if you've been here before,
 * you waltz right in. First-timers need to sign the guest book! 🎭
 */
function initAnnotatorName() {
    const name = getAnnotatorName();
    if (!name) {
        // Show the name prompt overlay
        document.getElementById('annotatorPromptOverlay').classList.add('open');
        // Auto-focus the input after a tiny delay (for animation)
        setTimeout(() => {
            document.getElementById('annotatorNameInput').focus();
        }, 200);
    }
}

/**
 * confirmAnnotatorName()
 *
 * Validates and saves the name from the prompt input.
 * Closes the overlay on success, shows error on empty.
 */
function confirmAnnotatorName() {
    const input = document.getElementById('annotatorNameInput');
    const errorEl = document.getElementById('annotatorNameError');
    const name = input.value.trim();

    if (!name) {
        errorEl.textContent = "Honey, we need a name! Even stage names count! 💅";
        errorEl.style.display = 'block';
        input.focus();
        return;
    }

    // Basic sanitization: alphanumeric + spaces + common chars
    if (name.length < 2) {
        errorEl.textContent = "That's too short, darling — at least 2 characters!";
        errorEl.style.display = 'block';
        input.focus();
        return;
    }

    if (name.length > 50) {
        errorEl.textContent = "Whoa, keep it under 50 characters, superstar! ✨";
        errorEl.style.display = 'block';
        input.focus();
        return;
    }

    setAnnotatorName(name);
    errorEl.style.display = 'none';
    document.getElementById('annotatorPromptOverlay').classList.remove('open');
    updateAnnotatorBadge();
    showToast(`👋 Welcome, ${name}! Your annotations will be tracked. ✨`, 'success', 4000);
}

/**
 * updateAnnotatorBadge()
 * Updates the 👤 badge in the header with the current annotator name.
 */
function updateAnnotatorBadge() {
    const badge = document.getElementById('annotatorBadgeName');
    if (!badge) return;
    const name = getAnnotatorName();
    badge.textContent = name || '(set name)';
    const container = document.getElementById('annotatorBadge');
    if (container) {
        container.style.borderStyle = name ? 'solid' : 'dashed';
        container.style.color = name ? 'var(--accent-cyan)' : 'var(--text-muted)';
    }
}

/**
 * changeAnnotatorName()
 * Quick inline name change via prompt — click the 👤 badge to trigger.
 */
function changeAnnotatorName() {
    const current = getAnnotatorName();
    const newName = prompt('Annotator A (primary) name:', current || '');
    if (newName === null) return;
    if (newName.trim().length < 2) {
        showToast('Name needs at least 2 characters!', 'error', 3000);
        return;
    }
    // ── Guard: can't match Annotator B if IAA is active ──────────
    const iaaName = localStorage.getItem('iaa_annotator_name') || '';
    if (iaaMode && iaaName && newName.trim().toLowerCase() === iaaName.toLowerCase()) {
        showToast(
            `❌ "${newName.trim()}" is already Annotator B! ` +
            `A and B must be different people! 👯`,
            'error', 5000
        );
        return;
    }
    setAnnotatorName(newName.trim());
    updateAnnotatorBadge();
    showToast(`👤 Annotator A changed to "${newName.trim()}" ✨`, 'success', 3000);
}

/**
 * changeIAAAnnotatorName()
 *
 * 📏 Change Annotator B's name while IAA mode is active.
 * Click the name on the pink IAA banner to trigger.
 *
 * Guards:
 *   - Must be different from Annotator A (the whole point of IAA!)
 *   - At least 2 characters
 *   - Updates banner display immediately
 *
 * 💅 Like swapping out a drag name mid-show — the performance
 * continues, just under a new spotlight! 🌟
 */
function changeIAAAnnotatorName() {
    const currentB = localStorage.getItem('iaa_annotator_name') || '';
    const primaryA = getAnnotatorName() || '(unknown)';

    const newName = prompt(
        `Change Annotator B name.

` +
        `Annotator A (primary): "${primaryA}"
` +
        `Annotator B (current): "${currentB}"

` +
        `Enter new name for Annotator B:`,
        currentB
    );

    if (newName === null) return;  // Cancelled
    if (newName.trim().length < 2) {
        showToast('Name needs at least 2 characters!', 'error', 3000);
        return;
    }

    // ── Guard: B must NOT equal A ────────────────────────────────
    if (newName.trim().toLowerCase() === primaryA.toLowerCase()) {
        showToast(
            `❌ "${newName.trim()}" is Annotator A! ` +
            `IAA needs a DIFFERENT person — two judges, not one! 👯`,
            'error', 5000
        );
        return;
    }

    // ── Apply the change ─────────────────────────────────────────
    localStorage.setItem('iaa_annotator_name', newName.trim());
    document.getElementById('iaaBannerName').textContent = newName.trim();
    showToast(`📏 Annotator B changed to "${newName.trim()}" ✨`, 'success', 3000);
}

// -----------------------------------------------------------------
// 📏 IAA MODE — Toggle between primary and IAA annotation
// -----------------------------------------------------------------

/** Stash of Annotator A's annotations — restored on exit */
let primaryAnnotationsBackup = null;

/**
 * enterIAAMode()
 *
 * THE FULL TRANSFORMATION:
 *   1. Backs up & HIDES Annotator A's annotations (clean slate, no bias)
 *   2. Adds body.iaa-active (pink theme overrides kick in)
 *   3. Dims normal Save/Complete buttons (use Save IAA instead)
 *   4. Shows the IAA banner (between toolbar and layout — VISIBLE!)
 *   5. Stores state in localStorage (persists across candidates)
 */
function enterIAAMode() {
    // ── IAA needs a SEPARATE Annotator B name ─────────────────────
    // The primary annotator is already saved with the doc.
    // IAA compares TWO DIFFERENT people — so we need a DIFFERENT name!
    // We store it in its own localStorage key to avoid conflicts. 💅
    let annotatorName = localStorage.getItem('iaa_annotator_name') || '';

    if (!annotatorName) {
        const primaryName = getAnnotatorName() || '(unknown)';
        const input = prompt(
            `📏 IAA Mode: Enter ANNOTATOR B name.\n\n` +
            `Primary annotator (A) is "${primaryName}".\n` +
            `You must use a DIFFERENT name for the second annotator!\n\n` +
            `Example: If A is "Soraya", B could be "Ahmad" or "Reviewer2".`
        );
        if (!input || input.trim().length < 2) {
            showToast('Need a name (2+ chars) to activate IAA!', 'error', 4000);
            return;
        }
        // ── Guard: Annotator B must NOT be Annotator A ───────────
        if (input.trim().toLowerCase() === (getAnnotatorName() || '').toLowerCase()) {
            showToast(
                `❌ "${input.trim()}" is your primary annotator name! ` +
                `IAA needs a DIFFERENT person — two judges, not one judge twice! 👯`,
                'error', 6000
            );
            return;
        }
        annotatorName = input.trim();
        localStorage.setItem('iaa_annotator_name', annotatorName);
    }

    iaaMode = true;

    // ── Backup & clear primary annotations ───────────────────────────
    // We always back up Annotator A's work so we can restore it on exit.
    primaryAnnotationsBackup = JSON.parse(JSON.stringify(annotations));
    annotations = [];
    renderText();
    updateAnnoList();

    // ── Visual transformation ─────────────────────────────────────────
    document.body.classList.add('iaa-active');
    const normalBtns = document.getElementById('normalSaveButtons');
    if (normalBtns) normalBtns.classList.add('iaa-dimmed-btn');

    // ── Show banner ───────────────────────────────────────────────────
    const banner = document.getElementById('iaaBanner');
    banner.style.display = 'flex';
    document.getElementById('iaaBannerName').textContent = annotatorName;

    // Persist IAA mode across page loads / candidate navigation
    localStorage.setItem('iaa_mode_active', 'true');

    // ── THE FIX: Load previously saved IAA annotations for this doc ───
    // Without this, every time Annotator B re-enters IAA mode their
    // previous labels are gone — like re-painting a canvas you already
    // finished! This fetches their saved work from the DB and restores it.
    setIAASaveStatus('loading');  // Show a loading indicator immediately
    fetch(`/api/iaa/annotations/${DOC_ID}?annotator=${encodeURIComponent(annotatorName)}`)
        .then(res => res.json())
        .then(data => {
            if (data.found && data.annotations && data.annotations.length > 0) {
                // ✅ Previous IAA work found — restore it!
                annotations = data.annotations;
                renderText();
                updateAnnoList();
                setIAASaveStatus('saved', data.span_count);
                showToast(
                    `📏 IAA Mode ON — restored ${data.span_count} saved annotation` +
                    `${data.span_count !== 1 ? 's' : ''} by "${annotatorName}". ` +
                    `Continue labeling or save when done! ✨`,
                    'success', 5000
                );
            } else {
                // 🆕 No previous work — clean slate as expected
                setIAASaveStatus('unsaved');
                showToast(
                    `📏 IAA Mode ON — annotating as "${annotatorName}". ` +
                    `Primary annotations hidden. Label from scratch! ✨`,
                    'success', 5000
                );
            }
        })
        .catch(err => {
            // Network error loading previous annotations — warn but don't block
            console.warn('Could not load previous IAA annotations:', err);
            setIAASaveStatus('unsaved');
            showToast(
                `📏 IAA Mode ON — annotating as "${annotatorName}". ` +
                `(Could not load previous work: ${err}) ⚠️`,
                'warning', 5000
            );
        });
}

/**
 * exitIAAMode()
 *
 * Restores everything: annotations, theme, buttons, banner.
 */
function exitIAAMode() {
    iaaMode = false;

    // Restore primary annotations
    if (primaryAnnotationsBackup) {
        annotations = primaryAnnotationsBackup;
        primaryAnnotationsBackup = null;
        renderText();
        updateAnnoList();
    }

    // Reverse visual transformation
    document.body.classList.remove('iaa-active');
    const normalBtns = document.getElementById('normalSaveButtons');
    if (normalBtns) normalBtns.classList.remove('iaa-dimmed-btn');
    document.getElementById('iaaBanner').style.display = 'none';

    localStorage.removeItem('iaa_mode_active');
    // 💅 Clear IAA annotator name so next IAA session prompts fresh
    localStorage.removeItem('iaa_annotator_name');

    showToast('📏 IAA Mode OFF — primary annotations restored! ✨', 'success', 3000);
}

/**
 * autoDetectIAAMode()
 *
 * Called on page load. If IAA was activated from the queue page,
 * auto-enters IAA mode so Annotator B gets a clean slate
 * on every candidate without clicking anything extra.
 */
function autoDetectIAAMode() {
    if (localStorage.getItem('iaa_mode_active') === 'true') {
        setTimeout(() => enterIAAMode(), 300);
    }
}

/**
 * saveIAAAnnotations()
 *
 * Saves the current annotation state as Annotator B's IAA set.
 * Routes to /api/iaa/save with the annotator name attached.
 *
 * Edge case: If annotations array is empty, warns and bails.
 */
function saveIAAAnnotations() {
    // ── Use the IAA-specific name, NOT the primary annotator name ──
    const annotatorName = localStorage.getItem('iaa_annotator_name') || '';
    if (!annotatorName) {
        showToast('❌ No annotator name set — please enter your name first!', 'error', 4000);
        return;
    }

    if (!annotations || annotations.length === 0) {
        showToast('❌ No annotations to save — label some entities first! 🎨', 'error', 4000);
        return;
    }

    // Show saving state on the banner button
    setIAASaveStatus('saving');

    const payload = {
        doc_id: DOC_ID,
        annotator_name: annotatorName,
        annotations: annotations
    };

    fetch('/api/iaa/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => {
        if (!res.ok) {
            return res.json().catch(() => {
                throw new Error(`Server error ${res.status}`);
            }).then(errData => {
                throw new Error(errData.error || `HTTP ${res.status}`);
            });
        }
        return res.json();
    })
    .then(data => {
        if (data.success) {
            // ✅ Update the banner status to "saved" with span count
            setIAASaveStatus('saved', data.spans);
            showToast(
                `✅ IAA annotations saved! ${data.spans} span${data.spans !== 1 ? 's' : ''} ` +
                `by "${annotatorName}" — your labels are locked in, darling! 🏆`,
                'success', 5000
            );
        } else {
            setIAASaveStatus('error');
            showToast('❌ IAA save failed: ' + (data.error || 'Unknown'), 'error', 6000);
        }
    })
    .catch(err => {
        setIAASaveStatus('error');
        showToast('❌ Network error saving IAA: ' + err, 'error', 6000);
    });
}


// -----------------------------------------------------------------
// 📏📊 IAA DASHBOARD — Modal with heatmap + metrics
// -----------------------------------------------------------------

/**
 * openIAAModal()
 *
 * Opens the IAA dashboard modal. On first open, fetches the annotator
 * list so the selector is populated before any data loads.
 */
function openIAAModal() {
    openInspector('iaa');
    // Fetch annotators first, then load the dashboard data.
    // Like getting the guest list before the party starts! 🎉
    loadIAAAnnotators().then(() => loadIAADashboard());
}

/**
 * loadIAAAnnotators()
 *
 * Fetches the list of all Annotator B names from the server and
 * refreshes the annotator selector UI in the modal header.
 * Returns a Promise so callers can chain .then().
 */
function loadIAAAnnotators() {
    return fetch('/api/iaa/annotators')
        .then(res => res.ok ? res.json() : { annotators: [] })
        .then(data => {
            iaaAnnotatorList = data.annotators || [];
            renderAnnotatorSelector();
        })
        .catch(() => {
            iaaAnnotatorList = [];
            renderAnnotatorSelector();
        });
}

/**
 * renderAnnotatorSelector()
 *
 * Builds the annotator filter pill-buttons in the IAA modal header.
 * Shows an "All" pill plus one pill per unique Annotator B, each
 * showing name + doc count. Active pill is highlighted pink.
 *
 * Like a guest list at the door — tap a name to see only their scores! 💅
 */
function renderAnnotatorSelector() {
    const container = document.getElementById('iaaAnnotatorSelector');
    if (!container) return;

    if (iaaAnnotatorList.length === 0) {
        // No IAA annotators yet — hide the selector entirely
        container.innerHTML = '';
        container.style.display = 'none';
        return;
    }

    container.style.display = 'flex';

    // Build pill buttons: "All" + one per annotator
    const allActive = iaaSelectedAnnotator === null;
    let html = `
        <!-- "All Annotators" pill -->
        <button class="iaa-annotator-pill ${allActive ? 'active' : ''}"
                onclick="selectIAAAnnotator(null)"
                title="Show aggregate across all annotators">
            👥 All
            <span class="iaa-annotator-pill-count">${iaaAnnotatorList.reduce((s, a) => s + (a.doc_count || 0), 0)} docs</span>
        </button>`;

    for (const ann of iaaAnnotatorList) {
        const isActive = iaaSelectedAnnotator === ann.annotator_name;
        // Format last annotated date nicely (just show date, not time)
        const lastDate = ann.last_annotated
            ? ann.last_annotated.split(' ')[0]
            : '';
        html += `
        <button class="iaa-annotator-pill ${isActive ? 'active' : ''}"
                onclick="selectIAAAnnotator(${JSON.stringify(ann.annotator_name)})"
                title="Last annotated: ${escapeHtml(lastDate)}">
            👤 ${escapeHtml(ann.annotator_name)}
            <span class="iaa-annotator-pill-count">${ann.doc_count} doc${ann.doc_count !== 1 ? 's' : ''}</span>
        </button>`;
    }

    // "Add annotator" mini-input — lets you register a new IAA annotator name
    // so they can enter IAA mode even before they've saved their first doc
    html += `
        <button class="iaa-annotator-pill iaa-annotator-add-btn"
                onclick="showAddAnnotatorPrompt()"
                title="Add a new IAA annotator">
            ＋ Add
        </button>`;

    container.innerHTML = html;
}

/**
 * selectIAAAnnotator(name)
 *
 * Sets the active annotator filter and reloads the dashboard.
 * Passing null shows all annotators aggregated.
 */
function selectIAAAnnotator(name) {
    iaaSelectedAnnotator = name;
    renderAnnotatorSelector();   // Update active pill immediately
    if (iaaDashboardScope === 'all') {
        loadIAADashboard();          // Reload with new filter
    }
}

/**
 * showAddAnnotatorPrompt()
 *
 * Shows a tiny inline input to type a new annotator name.
 * On confirm, sets that person as the active annotator so they
 * can immediately enter IAA mode — their name will appear in the
 * selector once they've actually saved their first annotation.
 */
function showAddAnnotatorPrompt() {
    const container = document.getElementById('iaaAnnotatorSelector');
    if (!container) return;

    // Replace the ＋ Add button with an inline input form
    const addBtn = container.querySelector('.iaa-annotator-add-btn');
    if (!addBtn) return;

    addBtn.outerHTML = `
        <span style="display:inline-flex; align-items:center; gap:4px;">
            <input id="iaaNewAnnotatorInput"
                   type="text"
                   placeholder="Annotator name…"
                   maxlength="60"
                   style="font-size:0.7rem; padding:3px 8px; border-radius:6px;
                           border:1px solid var(--accent-pink); background:var(--bg-elevated);
                           color:var(--text-primary); width:130px; outline:none;"
                   onkeydown="if(event.key==='Enter') confirmAddAnnotator();
                              if(event.key==='Escape') renderAnnotatorSelector();">
            <button class="iaa-annotator-pill active"
                    onclick="confirmAddAnnotator()"
                    style="padding:3px 8px;">✓</button>
            <button class="iaa-annotator-pill"
                    onclick="renderAnnotatorSelector()"
                    style="padding:3px 8px;">✕</button>
        </span>`;

    document.getElementById('iaaNewAnnotatorInput')?.focus();
}

/**
 * confirmAddAnnotator()
 *
 * Reads the new annotator name from the inline input, sets it as the
 * active annotator, and tells the user to enter IAA mode under that name.
 * The name will persist in the selector after they save their first doc.
 */
function confirmAddAnnotator() {
    const input = document.getElementById('iaaNewAnnotatorInput');
    const name = input ? input.value.trim() : '';
    if (!name) {
        renderAnnotatorSelector();
        return;
    }

    // Pre-select this annotator so the dashboard is ready to show their work
    iaaSelectedAnnotator = name;

    // Also pre-fill the annotator name input so entering IAA mode is seamless
    const nameInput = document.getElementById('annotatorNameInput');
    if (nameInput) {
        nameInput.value = name;
        // Persist to localStorage immediately
        localStorage.setItem('annotator_name', name);
    }

    renderAnnotatorSelector();
    showToast(
        `👤 "${name}" added! Close this dashboard and click 📏 Enter IAA Mode ` +
        `to start annotating. Their scores will appear here after saving. ✨`,
        'success', 6000
    );
}

/**
 * toggleIAAScope()
 *
 * Switches between 'This Doc' and 'All Docs' views in the dashboard.
 */
function toggleIAAScope() {
    const btn = document.getElementById('iaaToggleScope');
    if (iaaDashboardScope === 'all') {
        iaaDashboardScope = 'doc';
        btn.textContent = '📊 All Docs';
        btn.title = 'Show aggregate metrics across all documents';
    } else {
        iaaDashboardScope = 'all';
        btn.textContent = '🔍 This Doc';
        btn.title = 'Show metrics for this document only';
    }
    loadIAADashboard();
}

/**
 * loadIAADashboard()
 *
 * Fetches IAA data from the appropriate endpoint and renders it.
 * Respects the active scope (doc/all) and annotator filter.
 */
function loadIAADashboard() {
    const body = document.getElementById('iaaModalBody');
    body.innerHTML = '<div style="text-align:center; padding:40px; color:var(--text-muted);">Loading IAA metrics... ⏳</div>';

    let url;
    if (iaaDashboardScope === 'doc') {
        url = `/api/iaa/compute/${DOC_ID}`;
    } else {
        // Build URL with optional annotator filter query param
        url = iaaSelectedAnnotator
            ? `/api/iaa/dashboard?annotator=${encodeURIComponent(iaaSelectedAnnotator)}`
            : '/api/iaa/dashboard';
    }

    fetch(url)
        .then(res => {
            if (!res.ok) {
                return res.json().catch(() => {
                    throw new Error(`Server error ${res.status} (${res.statusText})`);
                }).then(errData => {
                    throw new Error(errData.error || errData.message || `HTTP ${res.status}`);
                });
            }
            return res.json();
        })
        .then(data => {
            if (data.status === 'error') {
                body.innerHTML = `<div style="text-align:center; padding:40px; color:var(--accent-red);">
                    ❌ IAA Error: ${escapeHtml(data.error || data.message || 'Unknown error')}<br>
                    <small style="color:var(--text-muted); margin-top:8px; display:block;">
                        Check the server logs for the full traceback.
                    </small></div>`;
            } else if (data.status === 'no_data') {
                renderIAAEmptyState(body, data.message);
            } else if (iaaDashboardScope === 'doc') {
                renderIAASingleDoc(body, data);
            } else {
                renderIAAAllDocs(body, data);
            }
        })
        .catch(err => {
            body.innerHTML = `<div style="text-align:center; padding:40px; color:var(--accent-red);">
                ❌ Failed to load IAA data: ${escapeHtml(String(err))}</div>`;
        });
}

/**
 * renderIAAEmptyState(container, message)
 *
 * Shows a friendly empty state when no IAA data exists yet.
 */
function renderIAAEmptyState(container, message) {
    container.innerHTML = `
        <div class="iaa-empty-state">
            <div class="iaa-empty-icon">📏</div>
            <h4>No IAA Data Yet</h4>
            <p>${escapeHtml(message || 'No documents have been dual-annotated yet.')}</p>
            <div style="margin-top:18px; font-size:0.72rem; color:var(--text-secondary); line-height:1.8;">
                <strong>How to start IAA:</strong><br>
                1. Have Annotator A complete annotations normally<br>
                2. A different annotator enters <strong>IAA Mode</strong> (📏 button)<br>
                3. Annotator B labels the same document independently<br>
                4. Click <strong>💾 Save IAA</strong> to store their labels<br>
                5. Come back here to see agreement metrics! 🏆
            </div>
            <div style="margin-top:16px;">
                <button class="btn btn-sm" onclick="closeInspector('iaa'); enterIAAMode();"
                        style="border-color:var(--accent-pink); color:var(--accent-pink); padding:6px 18px;">
                    📏 Enter IAA Mode
                </button>
            </div>
        </div>
    `;
}

/**
 * kappaInterpretation(k)
 *
 * Returns a human-readable label + CSS class for a Kappa value.
 * Uses Landis & Koch (1977) interpretation scale.
 *
 * Think of it as the Michelin star rating for agreement:
 *   ⭐⭐⭐ Almost Perfect  (0.81 - 1.00)
 *   ⭐⭐   Substantial     (0.61 - 0.80)
 *   ⭐     Moderate        (0.41 - 0.60)
 *   (meh)  Fair            (0.21 - 0.40)
 *   (yikes) Slight/Poor    (≤ 0.20)
 *
 * @param {number} k - Cohen's Kappa value (-1 to 1)
 * @returns {Object} label (string), cls (string)
 */
function kappaInterpretation(k) {
    if (k >= 0.81) return { label: 'Almost Perfect', cls: 'kappa-almost-perfect' };
    if (k >= 0.61) return { label: 'Substantial',    cls: 'kappa-substantial' };
    if (k >= 0.41) return { label: 'Moderate',       cls: 'kappa-moderate' };
    if (k >= 0.21) return { label: 'Fair',           cls: 'kappa-fair' };
    return { label: 'Poor/Slight', cls: 'kappa-poor' };
}

/**
 * f1Color(f1)
 *
 * Returns a CSS color for an F1 value — green for high, red for low.
 * Used for both heatmap cells and inline score displays.
 *
 * @param {number} f1 - F1 score (0.0 to 1.0)
 * @returns {string} CSS color string
 */
function f1Color(f1) {
    if (f1 >= 0.8)  return 'rgba(86,211,100,0.85)';   // Green — great!
    if (f1 >= 0.6)  return 'rgba(126,232,250,0.75)';   // Cyan — decent
    if (f1 >= 0.4)  return 'rgba(227,179,65,0.75)';    // Yellow — meh
    if (f1 >= 0.2)  return 'rgba(240,136,62,0.75)';    // Orange — concerning
    return 'rgba(218,54,51,0.7)';                       // Red — yikes
}

/**
 * f1Background(f1)
 *
 * Returns a lighter background color for heatmap cells.
 * Lower opacity than f1Color so text remains readable.
 *
 * @param {number} f1 - F1 score (0.0 to 1.0)
 * @returns {string} CSS background color
 */
function f1Background(f1) {
    if (f1 >= 0.8)  return 'rgba(86,211,100,0.2)';
    if (f1 >= 0.6)  return 'rgba(126,232,250,0.15)';
    if (f1 >= 0.4)  return 'rgba(227,179,65,0.15)';
    if (f1 >= 0.2)  return 'rgba(240,136,62,0.15)';
    return 'rgba(218,54,51,0.15)';
}

/**
 * renderIAASingleDoc(container, data)
 *
 * Renders IAA metrics for a single document.
 * Shows span-level and token-level agreement with per-entity breakdown.
 */
function renderIAASingleDoc(container, data) {
    const span = data.span_agreement || {};
    const kappa = data.token_kappa || {};
    const exact = span.exact || {};
    const partial = span.partial || {};
    const perEntity = span.per_entity || {};

    // KPI cards
    const kappaVal = kappa.kappa != null ? kappa.kappa : null;
    const kappaInterp = kappaVal != null ? kappaInterpretation(kappaVal) : null;

    let html = `
        <div style="font-size:0.72rem; color:var(--text-muted); margin-bottom:12px;">
            Comparing primary annotations vs <strong style="color:var(--accent-pink);">${escapeHtml(data.annotator_b || '?')}</strong>
            on document <strong>${escapeHtml(data.doc_id || '')}</strong>
        </div>

        <div class="iaa-kpi-row">
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${kappaVal != null ? f1Color(kappaVal) : 'var(--text-muted)'}">
                    ${kappaVal != null ? kappaVal.toFixed(3) : '—'}
                </div>
                <div class="iaa-kpi-label">Cohen's Kappa</div>
                ${kappaInterp ? `<div class="kappa-badge ${kappaInterp.cls}" style="margin-top:4px;">${kappaInterp.label}</div>` : ''}
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${f1Color(exact.f1 || 0)}">
                    ${(exact.f1 || 0).toFixed(3)}
                </div>
                <div class="iaa-kpi-label">Exact F1</div>
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${f1Color(partial.f1 || 0)}">
                    ${(partial.f1 || 0).toFixed(3)}
                </div>
                <div class="iaa-kpi-label">Partial F1</div>
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:var(--text-primary)">
                    ${kappa.tokens || 0}
                </div>
                <div class="iaa-kpi-label">Tokens Compared</div>
            </div>
        </div>
    `;

    // Per-entity breakdown as mini heatmap
    const entityNames = Object.keys(perEntity).sort();
    if (entityNames.length > 0) {
        html += buildEntityHeatmapSection(perEntity, entityNames, data);
    }

    container.innerHTML = html;
}

/**
 * renderIAAAllDocs(container, data)
 *
 * Renders aggregate IAA metrics across all dual-annotated documents.
 * Shows overall KPIs, per-entity heatmap, and per-document table.
 */
function renderIAAAllDocs(container, data) {
    const agg = data.aggregate || {};
    const docs = data.docs || [];
    const perEntity = agg.per_entity || {};
    const entityColors = data.entity_colors || {};
    const entityLabels = data.entity_labels || {};
    // Active filter passed back from the server
    const activeAnnotator = data.annotator_filter || null;

    // Overall KPIs
    const kappaVal = agg.avg_kappa != null ? agg.avg_kappa : null;
    const kappaInterp = kappaVal != null ? kappaInterpretation(kappaVal) : null;

    // Context line: shows who is being shown
    const contextLine = activeAnnotator
        ? `Showing <strong style="color:var(--accent-pink);">${escapeHtml(activeAnnotator)}</strong>'s annotations across
           <strong style="color:var(--accent-cyan);">${agg.total_docs || 0}</strong> document${(agg.total_docs || 0) !== 1 ? 's' : ''}`
        : `Aggregate across <strong style="color:var(--accent-cyan);">${agg.total_docs || 0}</strong> dual-annotated documents
           · <span style="color:var(--text-muted);">all annotators</span>`;

    let html = `
        <div style="font-size:0.72rem; color:var(--text-muted); margin-bottom:12px;">
            ${contextLine}
        </div>

        <div class="iaa-kpi-row">
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${kappaVal != null ? f1Color(kappaVal) : 'var(--text-muted)'}">
                    ${kappaVal != null ? kappaVal.toFixed(3) : '—'}
                </div>
                <div class="iaa-kpi-label">Avg Cohen's Kappa</div>
                ${kappaInterp ? `<div class="kappa-badge ${kappaInterp.cls}" style="margin-top:4px;">${kappaInterp.label}</div>` : ''}
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${f1Color(agg.avg_exact_f1 || 0)}">
                    ${(agg.avg_exact_f1 || 0).toFixed(3)}
                </div>
                <div class="iaa-kpi-label">Avg Exact F1</div>
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${f1Color(agg.avg_partial_f1 || 0)}">
                    ${(agg.avg_partial_f1 || 0).toFixed(3)}
                </div>
                <div class="iaa-kpi-label">Avg Partial F1</div>
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:var(--text-primary)">
                    ${agg.total_docs || 0}
                </div>
                <div class="iaa-kpi-label">Documents</div>
            </div>
        </div>
    `;

    // ── Per-Entity Heatmap ────────────────────────────────────────
    // The STAR of the show! 🌟 A color-coded grid showing F1, Precision,
    // and Recall for each entity type. Red = disagreement, Green = harmony.
    const entityNames = Object.keys(perEntity).sort();
    if (entityNames.length > 0) {
        const metrics = ['avg_f1', 'avg_precision', 'avg_recall'];
        const metricLabels = { avg_f1: 'F1', avg_precision: 'Prec', avg_recall: 'Rec' };

        html += `
            <div style="margin-bottom:6px;">
                <h4 style="font-family:var(--font-display); font-size:0.85rem; color:var(--text-primary); margin-bottom:4px;">
                    🔥 Agreement Heatmap by Entity Type
                </h4>
                <div style="font-size:0.62rem; color:var(--text-muted);">
                    Hover for details · 🟢 High agreement · 🔴 Low agreement
                </div>
            </div>
            <div class="iaa-heatmap" style="grid-template-columns: 140px repeat(${metrics.length}, 1fr);">
                <!-- Header row -->
                <div class="iaa-heatmap-header"></div>
                ${metrics.map(m => `<div class="iaa-heatmap-header">${metricLabels[m]}</div>`).join('')}

                <!-- Data rows -->
                ${entityNames.map(eName => {
                    const eData = perEntity[eName];
                    const label = entityLabels[eName] || eName;
                    const color = entityColors[eName] || '#8b949e';
                    return `
                        <div class="iaa-heatmap-row-label" title="${eName}">
                            <span style="display:inline-block; width:8px; height:8px; border-radius:2px;
                                         background:${color}; margin-right:5px; flex-shrink:0;"></span>
                            ${escapeHtml(label)}
                        </div>
                        ${metrics.map(m => {
                            const val = eData[m] || 0;
                            return `<div class="iaa-heatmap-cell"
                                         style="background:${f1Background(val)}; color:${f1Color(val)};"
                                         title="${label} — ${metricLabels[m]}: ${val.toFixed(3)} (${eData.doc_count || 0} docs)">
                                        ${val.toFixed(2)}
                                    </div>`;
                        }).join('')}
                    `;
                }).join('')}
            </div>
        `;
    }

    // ── Per-Document Table ─────────────────────────────────────────
    if (docs.length > 0) {
        html += `
            <h4 style="font-family:var(--font-display); font-size:0.85rem; color:var(--text-primary);
                        margin-top:20px; margin-bottom:8px;">
                📋 Per-Document Breakdown
            </h4>
            <div style="max-height:250px; overflow-y:auto; border:1px solid var(--border); border-radius:8px;">
                <table class="iaa-doc-table">
                    <thead>
                        <tr>
                            <th>Doc ID</th>
                            <th>Annotator B</th>
                            <th>Exact F1</th>
                            <th>Partial F1</th>
                            <th>Kappa</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${docs.map(doc => {
                            const sa = doc.span_agreement || {};
                            const ex = sa.exact || {};
                            const pa = sa.partial || {};
                            const tk = doc.token_kappa || {};
                            const kVal = tk.kappa != null ? tk.kappa : null;
                            const kInterp = kVal != null ? kappaInterpretation(kVal) : null;
                            return `
                                <tr>
                                    <td><a href="/annotate/${doc.candidate_id || 0}" style="color:var(--accent-cyan);">
                                        ${escapeHtml(doc.doc_id)}</a></td>
                                    <td>${escapeHtml(doc.annotator_b || '—')}</td>
                                    <td style="color:${f1Color(ex.f1 || 0)}">${(ex.f1 || 0).toFixed(3)}</td>
                                    <td style="color:${f1Color(pa.f1 || 0)}">${(pa.f1 || 0).toFixed(3)}</td>
                                    <td>
                                        ${kVal != null ? `<span style="color:${f1Color(kVal)}">${kVal.toFixed(3)}</span>` : '—'}
                                        ${kInterp ? `<span class="kappa-badge ${kInterp.cls}" style="margin-left:4px;">${kInterp.label}</span>` : ''}
                                    </td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    // ── IAA Mode CTA ──────────────────────────────────────────────
    html += `
        <div style="margin-top:18px; padding-top:14px; border-top:1px solid var(--border);
                     display:flex; gap:8px; align-items:center;">
            <button class="btn btn-sm" onclick="enterIAAMode(); closeInspector('iaa');"
                    style="border-color:var(--accent-pink); color:var(--accent-pink); padding:6px 16px;">
                📏 Enter IAA Mode
            </button>
            <span style="font-size:0.65rem; color:var(--text-muted);">
                Annotate this document as a second reviewer
            </span>
        </div>
    `;

    container.innerHTML = html;
}

/**
 * buildEntityHeatmapSection(perEntity, entityNames, data)
 *
 * Builds the per-entity heatmap section for a SINGLE document view.
 * Shows F1, Precision, and Recall for each entity type.
 *
 * @param {Object} perEntity - { ENTITY_NAME: { f1, precision, recall, count_a, count_b } }
 * @param {string[]} entityNames - Sorted entity type names
 * @param {Object} data - Full API response (for colors/labels)
 * @returns {string} HTML string
 */
function buildEntityHeatmapSection(perEntity, entityNames, data) {
    const entityColors = (data && data.entity_colors) || COLOR_MAP || {};
    const metrics = ['f1', 'precision', 'recall'];
    const metricLabels = { f1: 'F1', precision: 'Prec', recall: 'Rec' };

    let html = `
        <div style="margin-bottom:6px;">
            <h4 style="font-family:var(--font-display); font-size:0.85rem; color:var(--text-primary); margin-bottom:4px;">
                🔥 Agreement by Entity Type
            </h4>
            <div style="font-size:0.62rem; color:var(--text-muted);">
                Hover for span counts · 🟢 High · 🔴 Low
            </div>
        </div>
        <div class="iaa-heatmap" style="grid-template-columns: 140px repeat(${metrics.length}, 1fr);">
            <div class="iaa-heatmap-header"></div>
            ${metrics.map(m => `<div class="iaa-heatmap-header">${metricLabels[m]}</div>`).join('')}
            ${entityNames.map(eName => {
                const eData = perEntity[eName];
                const label = (SCHEMA.entities[eName] || {}).label || eName;
                const color = entityColors[eName] || '#8b949e';
                return `
                    <div class="iaa-heatmap-row-label" title="${eName}">
                        <span style="display:inline-block; width:8px; height:8px; border-radius:2px;
                                     background:${color}; margin-right:5px; flex-shrink:0;"></span>
                        ${escapeHtml(label)}
                    </div>
                    ${metrics.map(m => {
                        const val = eData[m] || 0;
                        const tooltip = `${label} ${metricLabels[m]}: ${val.toFixed(3)} (A:${eData.count_a||0} / B:${eData.count_b||0} spans)`;
                        return `<div class="iaa-heatmap-cell"
                                     style="background:${f1Background(val)}; color:${f1Color(val)};"
                                     title="${tooltip}">
                                    ${val.toFixed(2)}
                                </div>`;
                    }).join('')}
                `;
            }).join('')}
        </div>
    `;
    return html;
}

