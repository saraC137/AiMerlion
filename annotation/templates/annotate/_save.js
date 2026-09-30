// =================================================================
// 🪞 SAVE PREVIEW — "Check your look before the runway!" 💅
// =================================================================

/** Tracks which status the user intends to save as */
let pendingSaveStatus = null;

/**
 * showSavePreview(status)
 * 
 * THE MAIN ENTRY POINT — replaces direct saveAnnotations() calls.
 * Opens a preview modal, fetches validation from the backend,
 * and lets the annotator review everything before committing.
 * 
 * Like a dress rehearsal before the big show! 🎭✨
 * 
 * @param {string} status - 'in_progress' or 'completed'
 */
function showSavePreview(status) {
    pendingSaveStatus = status;

    // Open the modal with loading state
    const overlay = document.getElementById('previewOverlay');
    const body = document.getElementById('previewBody');
    const statusLabel = document.getElementById('previewStatusLabel');
    const confirmBtn = document.getElementById('previewConfirmBtn');

    // Set header label based on target status
    statusLabel.textContent = status === 'completed' ? 'Mark Complete' : 'Draft';
    statusLabel.style.color = status === 'completed' 
        ? 'var(--accent-green)' : 'var(--accent-cyan)';

    // Reset confirm button
    confirmBtn.disabled = false;
    confirmBtn.textContent = status === 'completed' 
        ? '✅ Confirm Complete' : '💾 Confirm Save';
    confirmBtn.className = status === 'completed' 
        ? 'btn btn-success' : 'btn btn-primary';

    // Show loading spinner
    body.innerHTML = `
        <div class="preview-loading">
            <div class="spinner"></div>
            Validating ${annotations.length} annotation${annotations.length !== 1 ? 's' : ''}... ✨
        </div>
    `;

    overlay.classList.add('open');

    // Listen for keyboard shortcuts inside modal
    document.addEventListener('keydown', previewKeyHandler);

    // Fetch validation from backend
    fetch(`/api/preview-before-save/${CANDIDATE_ID}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            annotations: annotations,
            status: status
        })
    })
    .then(r => {
        if (!r.ok) throw new Error(`Server returned ${r.status}`);
        return r.json();
    })
    .then(data => {
        renderPreviewContent(data, status);
    })
    .catch(err => {
        body.innerHTML = `
            <div class="preview-loading" style="color: var(--accent-red);">
                <div style="font-size: 2rem; margin-bottom: 12px;">❌</div>
                <div>Failed to validate: ${escapeHtml(err.message)}</div>
                <div style="font-size: 0.72rem; margin-top: 8px; color: var(--text-muted);">
                    You can still save directly — close this and use Ctrl+Shift+S to bypass preview.
                </div>
            </div>
        `;
    });
}

/**
 * renderPreviewContent(data, status)
 *
 * 🪞 THE GRAND REVEAL — Completely rewritten!
 *
 * NOW shows EVERY entity type from the schema as an editable field,
 * whether or not it has been annotated. Empty fields show a dashed
 * "Add..." input so the annotator can fill them in right here.
 *
 * Layout:
 *   1. Stats banner (compact)
 *   2. Warnings (if any)
 *   3. Classification row (Function / Industry dropdowns)
 *   4. Full entity form — ALL fields, ALL categories, ALL editable
 *
 * @param {Object} data   - Response from /api/preview-before-save
 * @param {string} status - Target save status ('in_progress' | 'completed')
 */
function renderPreviewContent(data, status) {
    const body       = document.getElementById('previewBody');
    const confirmBtn = document.getElementById('previewConfirmBtn');
    const footerInfo = document.getElementById('previewFooterInfo');
    const s          = data.stats;
    let html         = '';

    // ── 1. STATS BANNER ──────────────────────────────────────────────
    // Compact row of KPI cards — IDs on total so pfMarkDelete() can update it live
    html += `<div class="preview-stats">
        <div class="preview-stat-card">
            <div class="stat-number" style="color:var(--accent-cyan);" id="prevStatTotal">${s.total_annotations}</div>
            <div class="stat-label">Total Labels</div>
        </div>
        <div class="preview-stat-card">
            <div class="stat-number" style="color:var(--accent-green);">${s.entity_type_count}</div>
            <div class="stat-label">Types Used</div>
        </div>
        <div class="preview-stat-card">
            <div class="stat-number" style="color:var(--accent-purple);">${s.human_count}</div>
            <div class="stat-label">Human 👤</div>
        </div>
        <div class="preview-stat-card">
            <div class="stat-number" style="color:var(--accent-orange);">${s.auto_count}</div>
            <div class="stat-label">Auto 🤖</div>
        </div>
        <div class="preview-stat-card">
            <div class="stat-number"
                 style="color:${(s.critical_errors + s.warnings) > 0 ? 'var(--accent-red)' : 'var(--accent-green)'};">
                ${s.critical_errors + s.warnings}
            </div>
            <div class="stat-label">Issues</div>
        </div>
    </div>`;

    // ── 2. WARNINGS ──────────────────────────────────────────────────
    // Sort: critical first, then warning, then info
    if (data.warnings && data.warnings.length > 0) {
        html += `<div class="preview-warnings">`;
        const sortOrder = { critical: 0, warning: 1, info: 2 };
        const sortedWarnings = [...data.warnings].sort((a, b) =>
            (sortOrder[a.level] || 9) - (sortOrder[b.level] || 9)
        );
        for (const w of sortedWarnings) {
            html += `<div class="preview-warning-item ${escapeHtml(w.level)}">
                <span class="preview-warning-icon">${w.icon || '⚠️'}</span>
                <span>${escapeHtml(w.message)}</span>
            </div>`;
        }
        html += `</div>`;
    } else {
        html += `<div class="preview-all-clear">
            ✅ All checks passed — you're ready to save! 🎉
        </div>`;
    }

    // ── 3. BUILD EXISTING-ANNOTATION LOOKUP ──────────────────────────
    // Map: entity_type → [{text, idx}, ...]  from the backend summary.
    // This tells us which types already have annotations vs. which are empty.
    //
    // Like the guest list for the gala — some seats already filled,
    // some still open for walk-ins! 🎟️
    const annoByType = {};
    if (data.summary) {
        for (const entityList of Object.values(data.summary)) {
            for (const entity of entityList) {
                annoByType[entity.entity_type] = (entity.texts || []).map(item =>
                    typeof item === 'object'
                        ? { text: item.text, idx: item.idx }
                        : { text: item,      idx: -1 }
                );
            }
        }
    }

    // ── 4. CLASSIFICATION (Function / Industry / Skills / Tags) ─────────
    // These two get dropdown selects, not plain text inputs,
    // because they map to a controlled vocabulary list. 💅
    // 🆕 Skills & Tags shown as read-only chip displays in preview.
    const cls      = data.classification || {};
    const funcVal  = cls.function || '';
    const indVal   = cls.industry  || '';

    const funcOptions = FUNCTION_OPTIONS.map(f =>
        `<option value="${escapeHtml(f)}" ${f === funcVal ? 'selected' : ''}>${escapeHtml(f)}</option>`
    ).join('');
    const indOptions = INDUSTRY_OPTIONS.map(i =>
        `<option value="${escapeHtml(i)}" ${i === indVal ? 'selected' : ''}>${escapeHtml(i)}</option>`
    ).join('');

    // 🆕 Build chip HTML for skills & tags in preview
    const previewHardSkills = cls.hard_skills || currentHardSkills || [];
    const previewSoftSkills = cls.soft_skills || currentSoftSkills || [];
    const previewTags = cls.tags || currentTags || [];

    const hardChips = previewHardSkills.map(s =>
        `<span class="skill-tag-chip">${escapeHtml(s)}</span>`
    ).join('') || '<span style="color:var(--text-muted);font-size:0.6rem;font-style:italic;">None</span>';

    const softChips = previewSoftSkills.map(s =>
        `<span class="skill-tag-chip soft">${escapeHtml(s)}</span>`
    ).join('') || '<span style="color:var(--text-muted);font-size:0.6rem;font-style:italic;">None</span>';

    const tagChips = previewTags.map(s =>
        `<span class="skill-tag-chip ai-tag">${escapeHtml(s)}</span>`
    ).join('') || '<span style="color:var(--text-muted);font-size:0.6rem;font-style:italic;">None</span>';

    html += `<div class="pf-classify">
        <div class="pf-classify-field">
            <label>🧠 Function / Department</label>
            <select id="previewFunctionSelect" class="pf-select">
                <option value="">— Select Function —</option>
                ${funcOptions}
            </select>
        </div>
        <div class="pf-classify-field">
            <label>🏢 Industry Sector</label>
            <select id="previewIndustrySelect" class="pf-select">
                <option value="">— Select Industry —</option>
                ${indOptions}
            </select>
        </div>
    </div>
    <div style="margin-top:8px;">
        <label style="font-size:0.62rem;color:var(--text-muted);display:block;margin-bottom:3px;">🔧 Hard Skills</label>
        <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">${hardChips}</div>
        <label style="font-size:0.62rem;color:var(--text-muted);display:block;margin-bottom:3px;">💬 Soft Skills</label>
        <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">${softChips}</div>
        <label style="font-size:0.62rem;color:var(--text-muted);display:block;margin-bottom:3px;">🏷️ Tags (AI)</label>
        <div style="display:flex;flex-wrap:wrap;gap:3px;">${tagChips}</div>
    </div>`;

    // ── 5. THE COMPLETE ENTITY FORM ───────────────────────────────────
    //
    // EVERY entity type from the schema is shown here — including empty ones.
    // Each field shows:
    //   • Existing annotation(s) → editable input + ✕ delete button
    //   • Always one dashed "Add…" input at the bottom for new values
    //
    // We skip:
    //   - FUNCTION / INDUSTRY  →  handled above in Classification section
    //   - SECTION_HEADER       →  structural text marker, not a profile field
    //   - BULLET_MARKER        →  structural text marker, not a profile field
    //
    // SUMMARY_TEXT lives in META in the schema but IS a profile field,
    // so we include it explicitly in the PROFESSIONAL section. 💅

    const SKIP_ENTITIES = new Set([
        'FUNCTION', 'INDUSTRY',       // handled in Classification above
        'SECTION_HEADER', 'BULLET_MARKER'  // structural META markers
    ]);

    // ── MULTI_VALUE_TYPES ─────────────────────────────────────────────
    //
    // Entity types where the user is allowed (and encouraged!) to enter
    // MORE THAN ONE value in the Save Preview form.
    //
    // Rules of thumb for what goes here:
    //   ✅ ADD if max_per_doc > 1 in the schema AND it makes real-world sense
    //      for a candidate to have multiple (e.g. two phone numbers, work email
    //      + personal email, past jobs, multiple skills)
    //   ❌ SKIP if it should only ever have one entry per person (name, DOB,
    //      NRIC, gender — there is only ONE you, darling! 💅)
    //
    // When a user fills the "Add…" input for a multi-value type, a brand-new
    // dashed "Add…" input instantly appears below it — no page refresh needed.
    // Think of it like handing someone a clipboard and they keep asking for
    // more lines to fill in! 📋✨
    // ══════════════════════════════════════════════════════════════════
    // 📝 LONG_TEXT_TYPES — Entities that need <textarea> instead of <input>
    //
    // 🐛 BUG FIX: Regular <input type="text"> cannot handle multi-line
    // content! When a user highlights a long job description paragraph
    // with newlines, the text would DISAPPEAR on save because:
    //   1. Newlines break the HTML value="..." attribute
    //   2. <input> elements literally can't store multi-line text
    //
    // These entity types commonly contain paragraph text, bullet points,
    // or multi-line descriptions — they get <textarea> elements instead!
    // Like giving a diva a proper stage instead of a broom closet! 🎭✨
    // ══════════════════════════════════════════════════════════════════
    const LONG_TEXT_TYPES = new Set([
        'JOB_DESCRIPTION',       // Multi-line job responsibilities
        'PROJECT_DESCRIPTION',   // Project details and achievements
        'SUMMARY_TEXT',          // Career summary / objective
        'METRIC',                // Can be detailed achievement text
    ]);

    const MULTI_VALUE_TYPES = new Set([
        // PERSONAL — multiple contact points are common
        'EMAIL',              // work + personal email
        'PHONE',              // mobile + office + home
        'LOCATION',           // home address + district area
        'EXPECTED_LOCATION',  // open to multiple cities

        // PROFESSIONAL — candidates have many roles
        'JOB_TITLE',          // current + previous titles
        'ORGANIZATION',       // current + all past employers
        'WORK_DATE',          // one range per role
        'WORK_LOCATION',      // different office locations
        'JOB_DESCRIPTION',    // many bullet points per role
        'METRIC',             // many quantitative achievements
        'PROJECT_TITLE',      // multiple projects
        'PROJECT_DESCRIPTION',// multiple project bullets
        'INDUSTRY',           // cross-sector experience

        // EDUCATION — multiple degrees are common
        'DEGREE',
        'INSTITUTION',
        'FIELD_OF_STUDY',
        'EDU_DATE',
        'GPA',

        // SKILLS — always multiple!
        'SKILL',
        'SOFT_SKILL',
        'CERTIFICATION',
        'LANGUAGE_SKILL',
        'PROFICIENCY',
        'CERT_ISSUER',
        'CERT_DATE',
    ]);

    // Category sections — controls grouping, icon, label, and field ORDER.
    // Order matters: most important fields appear at the top of each group.
    const CATEGORY_SECTIONS = [
        {
            label: '👤 Personal Info',
            // Required fields first, then demographics, then optional IDs
            types: [
                'PERSON_NAME', 'EMAIL', 'PHONE',
                'DATE_OF_BIRTH', 'GENDER', 'MARITAL_STATUS',
                'NATIONALITY', 'NRIC_ID',
                'LOCATION', 'EXPECTED_LOCATION'
            ]
        },
        {
            label: '💼 Professional Experience',
            // Current role first, then work history blocks, then summary
            types: [
                'JOB_TITLE', 'ORGANIZATION', 'WORK_DATE', 'WORK_LOCATION',
                'JOB_DESCRIPTION', 'METRIC',
                'PROJECT_TITLE', 'PROJECT_DESCRIPTION',
                'SUMMARY_TEXT'
            ]
        },
        {
            label: '🎓 Education',
            types: [
                'DEGREE', 'INSTITUTION', 'FIELD_OF_STUDY', 'EDU_DATE', 'GPA'
            ]
        },
        {
            label: '🛠️ Skills & Certifications',
            types: [
                'LANGUAGE_SKILL', 'PROFICIENCY',
                'SKILL', 'SOFT_SKILL', 'SKILL_CATEGORY',
                'CERTIFICATION', 'CERT_ISSUER', 'CERT_DATE'
            ]
        }
    ];

    html += `<div class="preview-form">`;

    for (const section of CATEGORY_SECTIONS) {
        // Filter to entity types that actually exist in this schema build
        // (future-proofs against schema additions/removals)
        const validTypes = section.types.filter(t => SCHEMA.entities[t] && !SKIP_ENTITIES.has(t));
        if (validTypes.length === 0) continue;

        html += `<div class="pf-section">
            <div class="pf-section-header">${escapeHtml(section.label)}</div>`;

        for (const entityType of validTypes) {
            const def        = SCHEMA.entities[entityType];
            const existing   = annoByType[entityType] || [];
            const color      = def.color || '#7ee8fa';
            const isRequired = !!def.is_required;
            const hasMissing = isRequired && existing.length === 0;

            html += `<div class="pf-field" data-entity-type="${entityType}">
                <!-- Label column: coloured dot + name + required marker + "Missing" badge -->
                <div class="pf-label">
                    <span class="pf-dot" style="background:${color};"></span>
                    <span>${escapeHtml(def.label)}</span>
                    ${isRequired
                        ? `<span class="pf-required" title="Required — must have at least one annotation">*</span>`
                        : ''}
                    ${hasMissing
                        ? `<span class="pf-missing-badge">Missing</span>`
                        : ''}
                </div>

                <!-- Inputs column: existing rows + always-visible "Add…" row -->
                <div class="pf-inputs" id="pf_inputs_${entityType}">`;

            // ── Render each existing annotation as an editable row ─────────
            // 🐛 BUG FIX: Use <textarea> for LONG_TEXT_TYPES to handle
            // multi-line content (job descriptions, summaries, etc.)
            // Regular <input> elements truncate text at newlines! 💀
            for (const ann of existing) {
                const isLongText = LONG_TEXT_TYPES.has(entityType);
                const safeText = escapeHtml(ann.text);
                
                if (isLongText) {
                    // 📝 TEXTAREA for long-form content
                    html += `<div class="pf-input-row"
                                  data-anno-idx="${ann.idx}"
                                  data-entity-type="${entityType}">
                        <textarea class="pf-input pf-textarea preview-edit-input"
                               data-anno-idx="${ann.idx}"
                               data-entity-type="${entityType}"
                               data-original="${safeText}"
                               oninput="this.classList.toggle('edited', this.value.trim() !== decodeHtmlEntities(this.dataset.original).trim())"
                               placeholder="${escapeHtml(def.label)}…"
                               title="Edit ${escapeHtml(def.label)}">${safeText}</textarea>
                        <button class="pf-del-btn"
                                onclick="pfMarkDelete(this, ${ann.idx}, '${entityType}')"
                                title="Mark for deletion">✕</button>
                    </div>`;
                } else {
                    // 📝 Regular INPUT for single-line content
                    html += `<div class="pf-input-row"
                                  data-anno-idx="${ann.idx}"
                                  data-entity-type="${entityType}">
                        <input type="text"
                               class="pf-input preview-edit-input"
                               value="${safeText}"
                               data-anno-idx="${ann.idx}"
                               data-entity-type="${entityType}"
                               data-original="${safeText}"
                               oninput="this.classList.toggle('edited', this.value !== this.dataset.original)"
                               placeholder="${escapeHtml(def.label)}…"
                               title="Edit ${escapeHtml(def.label)}"
                        />
                        <button class="pf-del-btn"
                                onclick="pfMarkDelete(this, ${ann.idx}, '${entityType}')"
                                title="Mark for deletion">✕</button>
                    </div>`;
                }
            }

            // ── Always render one dashed "Add…" input row ──────────────────
            //
            // For MULTI_VALUE_TYPES: when the user fills this input, a NEW
            // dashed row automatically appears below via pfHandleAddInput().
            // That spawned row ALSO gets the same handler — it's recursive!
            //
            // For single-value types: fills once and stops. No new row spawned.
            //
            // Like a pad of paper — tear off a sheet, another appears! 📋✨
            const isMulti = MULTI_VALUE_TYPES.has(entityType);
            const isLongText = LONG_TEXT_TYPES.has(entityType);

            // 🐛 BUG FIX: Use <textarea> for LONG_TEXT_TYPES "Add..." rows too!
            if (isLongText) {
                html += `<div class="pf-input-row pf-add-row"
                              data-anno-idx="-1"
                              data-entity-type="${entityType}"
                              data-is-new="true">
                    <textarea class="pf-input pf-textarea pf-add-input"
                           data-anno-idx="-1"
                           data-entity-type="${entityType}"
                           data-original=""
                           data-is-new="true"
                           data-is-multi="${isMulti}"
                           data-is-long-text="true"
                           oninput="pfHandleAddInput(this)"
                           placeholder="Add ${escapeHtml(def.label)}…"
                           title="Type to add a new ${escapeHtml(def.label)}"></textarea>
                </div>`;
            } else {
                html += `<div class="pf-input-row pf-add-row"
                              data-anno-idx="-1"
                              data-entity-type="${entityType}"
                              data-is-new="true">
                    <input type="text"
                           class="pf-input pf-add-input"
                           value=""
                           data-anno-idx="-1"
                           data-entity-type="${entityType}"
                           data-original=""
                           data-is-new="true"
                           data-is-multi="${isMulti}"
                           oninput="pfHandleAddInput(this)"
                           placeholder="Add ${escapeHtml(def.label)}…"
                           title="Type to add a new ${escapeHtml(def.label)}"
                    />
                </div>`;
            }

            html += `    </div><!-- /pf-inputs -->
            </div><!-- /pf-field -->`;
        }

        html += `</div><!-- /pf-section -->`;
    }

    html += `</div><!-- /preview-form -->`;

    // ── Inject into modal ──────────────────────────────────────────────
    body.innerHTML = html;

    // ── 6. CONFIRM BUTTON STATE ───────────────────────────────────────
    // Hard block: zero annotations → can't save as complete
    // Soft criticals: missing fields but still allowed with a warning
    // All clear: green light! 🟢
    if (!data.can_complete && status === 'completed') {
        confirmBtn.disabled    = true;
        confirmBtn.textContent = '🚫 Add Annotations First';
        confirmBtn.className   = 'btn';
        confirmBtn.style.opacity = '0.5';
        confirmBtn.style.cursor  = 'not-allowed';
        footerInfo.innerHTML = `
            <span style="color:var(--accent-red);">
                🚨 No annotations found — add some labels before marking complete.
            </span>`;

    } else if (status === 'completed' && s.critical_errors > 0) {
        confirmBtn.disabled    = false;
        confirmBtn.style.opacity = '1';
        confirmBtn.style.cursor  = 'pointer';
        footerInfo.innerHTML = `
            <span style="color:var(--accent-orange);">
                ⚠️ ${s.critical_errors} field${s.critical_errors !== 1 ? 's' : ''} flagged —
                resume may genuinely be missing them.
                You can still mark complete if you're sure. 👀
            </span>`;

    } else {
        confirmBtn.disabled    = false;
        confirmBtn.style.opacity = '1';
        confirmBtn.style.cursor  = 'pointer';
        footerInfo.innerHTML = `Press <strong>Enter</strong> to confirm, <strong>Escape</strong> to cancel`;
    }
}

/**
 * pfMarkDelete(btn, annoIdx, entityType)
 *
 * Toggles an existing annotation row as "marked for deletion" in the preview.
 * Updates the Total Labels live counter in the stats banner.
 * The ✕ button toggles to ↩ (undo) so nothing is irrevocable! 💅
 *
 * @param {HTMLElement} btn       - The delete button that was clicked
 * @param {number}      annoIdx   - Index in the annotations[] array
 * @param {string}      entityType - Entity type string (for logging)
 */
function pfMarkDelete(btn, annoIdx, entityType) {
    // Walk up to the parent .pf-input-row
    const row = btn.closest('.pf-input-row');
    if (!row) return;

    // Toggle deleted state
    const isDeleted = row.classList.toggle('deleted');
    btn.textContent = isDeleted ? '↩' : '✕';
    btn.title       = isDeleted ? 'Undo deletion' : 'Mark for deletion';
    btn.classList.toggle('undo', isDeleted);   // Green style when showing undo

    // ── Live-update Total Labels counter ─────────────────────────────
    // Count only real annotation rows (not .pf-add-row) that are not deleted
    const totalActive = document.querySelectorAll(
        '.pf-input-row:not(.deleted):not(.pf-add-row)'
    ).length;
    const totalEl = document.getElementById('prevStatTotal');
    if (totalEl) totalEl.textContent = totalActive;

    console.log(`${isDeleted ? '🗑️ Mark delete' : '↩ Undo delete'}: [${annoIdx}] ${entityType}`);
}

/**
 * pfHandleAddInput(input)
 *
 * Called by oninput on every "Add…" dashed field in the preview form.
 *
 * Does two things:
 *   1. Toggles CSS classes so the input turns from dashed/italic (empty)
 *      to solid/normal (has text) — pure visual feedback.
 *
 *   2. For MULTI_VALUE_TYPES only: when THIS input gets text AND it's the
 *      last "Add…" row for its entity type, instantly injects a new empty
 *      "Add…" row below it. Like summoning a fresh line on a form! 📋
 *
 * We never spawn a new row if the user hasn't typed anything — that would
 * give us an infinite cascade of empty rows. Very un-glamorous. 🙅
 *
 * @param {HTMLInputElement} input - The input that fired the event
 */
function pfHandleAddInput(input) {
    const hasText    = input.value.trim() !== '';
    const entityType = input.dataset.entityType;
    const isMulti    = input.dataset.isMulti === 'true';

    // ── Step 1: Toggle visual state ──────────────────────────────────
    // .edited       → yellow border when the user has changed the value
    // .pf-add-input → dashed/italic style when empty; removed when filled
    input.classList.toggle('edited',       hasText);
    input.classList.toggle('pf-add-input', !hasText);

    // ── Step 2: Spawn a new "Add…" row for multi-value types ─────────
    // Guard conditions — ALL must be true to spawn:
    //   a) Entity type allows multiple values
    //   b) User has actually typed something (not just focused / cleared)
    //   c) This input has NOT already triggered a spawn (data-spawned flag)
    //      ← THIS is the key fix for the "jumps every keypress" bug! 🐛
    //   d) This input IS the current last "Add…" row for its entity type
    //
    // Without guard (c), every single keypress would check "am I last?" —
    // and after the first spawn the new row IS last, so nothing happens,
    // but without the flag the OLD row still thinks it CAN spawn again
    // if the user clears and re-types. The flag makes it one-shot. 💅
    if (!isMulti || !hasText) return;

    // ── Guard (c): only spawn once per input ─────────────────────────
    // We stamp a custom data attribute the first time we spawn.
    // Subsequent keystrokes in the same input skip the spawn block. 🎯
    if (input.dataset.spawned === 'true') return;

    const inputsContainer = input.closest('.pf-inputs');
    if (!inputsContainer) return;

    // ── Guard (d): only spawn from the LAST add-row ──────────────────
    const allAddRows = inputsContainer.querySelectorAll('.pf-add-row');
    const lastAddRow = allAddRows[allAddRows.length - 1];
    if (input.closest('.pf-add-row') !== lastAddRow) return;

    // ── Mark this input as "already spawned" before we do anything ───
    // Must set BEFORE DOM manipulation so re-entrant oninput calls
    // (which can fire during insertAdjacentElement in some browsers) skip. 🔒
    input.dataset.spawned = 'true';

    // Retrieve entity label from schema for placeholder text
    const def   = SCHEMA.entities[entityType];
    const label = def ? def.label : entityType;

    // ── Build the new empty "Add…" row ───────────────────────────────
    // We create DOM elements directly (not innerHTML concat) so the event
    // handler binds correctly without needing eval(). Safer! 💅
    const newRow = document.createElement('div');
    newRow.className          = 'pf-input-row pf-add-row';
    newRow.dataset.annoIdx    = '-1';
    newRow.dataset.entityType = entityType;
    newRow.dataset.isNew      = 'true';

    // 🐛 BUG FIX: Create textarea for LONG_TEXT_TYPES, input for others
    const isLongText = input.dataset.isLongText === 'true' || LONG_TEXT_TYPES.has(entityType);
    const newInput = document.createElement(isLongText ? 'textarea' : 'input');
    
    if (!isLongText) {
        newInput.type = 'text';
        newInput.className = 'pf-input pf-add-input';
    } else {
        newInput.className = 'pf-input pf-textarea pf-add-input';
    }
    newInput.value       = '';
    newInput.placeholder = `Add another ${label}…`;
    newInput.title       = `Type to add another ${label}`;

    // Stamp all data attributes so confirmSave() picks this row up correctly
    newInput.dataset.annoIdx    = '-1';
    newInput.dataset.entityType = entityType;
    newInput.dataset.original   = '';
    newInput.dataset.isNew      = 'true';
    newInput.dataset.isMulti    = 'true';   // propagate multi flag to children
    newInput.dataset.spawned    = 'false';  // new row starts un-spawned

    // Attach the SAME handler — enables recursive multi-row addition 🔄
    newInput.addEventListener('input', () => pfHandleAddInput(newInput));

    newRow.appendChild(newInput);

    // Insert AFTER the current last add-row
    lastAddRow.insertAdjacentElement('afterend', newRow);

    // ── NO auto-focus ─────────────────────────────────────────────────
    // 🐛 BUG FIXED: auto-focus was here before.
    // It stole the cursor to the new blank row after the FIRST character,
    // mid-word. User now stays in their current input and can Tab or click
    // into the new row when they are done. 💅
}

/**
 * confirmSave(overrideStatus, shouldClose)
 * Collects data from the preview window and prepares it for the server.
 */
function confirmSave(overrideStatus = null, shouldClose = true) {
    const statusToSave = overrideStatus || pendingSaveStatus;

    // ── 🛡️ SAFETY NET — wrap everything in try-catch ─────────────
    // 🐛 BUG FIX: Previously, ANY error in data collection (e.g. an
    // out-of-bounds annotation index, a corrupted input value, or a
    // DOM element that vanished) would throw an uncaught TypeError.
    // That silently halted execution BEFORE saveAnnotations() ran,
    // so the user saw the preview close but nothing actually saved.
    // No toast, no alert, no error — just... silence. 😱
    //
    // Now we catch errors, show the user a clear message, and still
    // attempt the save with whatever data we managed to collect.
    // Like a stage manager calling "SHOW MUST GO ON!" even when
    // someone trips backstage! 🎭💪
    try {
        // Collect indices of annotations marked for deletion
        const deletedIndices = new Set();
        document.querySelectorAll('.pf-input-row.deleted').forEach(row => {
            const idx = row.dataset.annoIdx;
            if (idx && idx !== '-1') {
                deletedIndices.add(parseInt(idx, 10));
            }
        });

        // 🐛 BUG FIX: Include both input AND textarea elements!
        const allPreviewInputs = document.querySelectorAll('.pf-input, .pf-textarea');;
        allPreviewInputs.forEach(input => {
            const idx = parseInt(input.dataset.annoIdx, 10);
            const entityType = input.dataset.entityType;
            const isNew = input.dataset.isNew === 'true';
            const newText = input.value.trim();
            const row = input.closest('.pf-input-row');
            if (row && row.classList.contains('deleted')) return; // skip deleted

            if (!newText) return;

            if (!isNew && idx >= 0) {
                // 🛡️ Guard: verify the annotation at this index still exists
                // before updating. If the array was modified while the preview
                // was open (e.g. Delete key pressed outside input focus),
                // the index could be stale. Skip gracefully instead of crashing.
                if (idx < annotations.length && annotations[idx]) {
                    const oldText = annotations[idx].text;
                    
                    // 🐛 BUG FIX: Compare TRIMMED versions to avoid false positives
                    // from whitespace differences. The original highlight might have
                    // captured trailing spaces, but input.value.trim() removes them.
                    // Without this, unchanged annotations lose their char positions! 💅
                    const textActuallyChanged = (oldText || '').trim() !== newText;
                    
                    // Only update text if it actually changed
                    if (textActuallyChanged) {
                        annotations[idx].text = newText;

                        // 🐛 BUG FIX: When the user EDITS the text of an existing
                        // annotation, the char_start/char_end positions become STALE
                        // because they still point to the ORIGINAL text in RAW_TEXT.
                        //
                        // Before this fix, renderText() would re-slice RAW_TEXT at
                        // the old positions and show the ORIGINAL text — making it
                        // look like the edit was lost! 😱
                        //
                        // Now we mark edited annotations with char_start/char_end = -1
                        // to signal: "this is user-edited text — don't re-slice!"
                        // Think of it like crossing out a handwritten note and writing
                        // a new one — you can't pretend the old ink is still there! 🖊️✨
                        annotations[idx].char_start = -1;
                        annotations[idx].char_end   = -1;
                        annotations[idx].annotator  = 'human';  // Mark as human-edited
                        console.log(`✏️ Edited [${idx}] ${entityType}: "${oldText}" → "${newText}" (positions reset)`);
                    }
                } else {
                    console.warn(`⚠️ confirmSave: annotations[${idx}] is undefined — skipping update for ${entityType}`);
                }
            } else if (isNew && newText) {
                // Push new annotation (manual entry, no char positions)
                annotations.push({
                    entity_type: entityType,
                    char_start: -1,
                    char_end: -1,
                    text: newText,
                    layer: 0,
                    annotator: 'human',
                    confidence: 1.0
                });
            }
        });

        // Remove deleted annotations
        if (deletedIndices.size > 0) {
            annotations = annotations.filter((_, idx) => !deletedIndices.has(idx));
        }

        // Sync classification from preview to main selects
        const previewFunc = document.getElementById('previewFunctionSelect');
        const previewInd = document.getElementById('previewIndustrySelect');
        if (previewFunc && previewInd) {
            const funcSelect = document.getElementById('functionSelect');
            const indSelect = document.getElementById('industrySelect');
            if (funcSelect) funcSelect.value = previewFunc.value;
            if (indSelect) indSelect.value = previewInd.value;
        }
    } catch (err) {
        // ── Error during data collection — warn but don't abort ──────
        console.error('❌ confirmSave data collection error:', err);
        showToast(`⚠️ Some edits may not have been captured: ${err.message}`, 'warning');
        // Fall through to save whatever we DID manage to collect
    }

    // Close or refresh UI
    if (shouldClose) {
        closePreview();
    } else {
        showSavePreview(statusToSave); // re-render with updated data
    }

    // Actually save to server — always attempt this, even if collection had issues
    saveAnnotations(statusToSave);

    // Refresh text highlights and annotation list
    renderText();
    if (activeInspector === 'labels') openLabelsModal();
}

/**
 * markPreviewDelete(btn, annoIdx)
 * 
 * Toggles an annotation row as "marked for deletion" in the preview.
 * ALSO live-updates:
 *   - Per-entity-type count badges
 *   - Total Labels stat in the banner
 * Everything stays in sync — like a perfectly choreographed number! 🎭💅
 */
function markPreviewDelete(btn, annoIdx) {
    const row = btn.closest('.preview-edit-row');
    if (!row) return;

    const isDeleted = row.classList.toggle('deleted');
    btn.textContent = isDeleted ? '↩' : '✕';
    btn.title = isDeleted ? 'Undo deletion' : 'Mark for deletion';

    // ── Update per-entity-type count ─────────────────────────────
    const entityType = row.dataset.entityType;
    if (entityType) {
        const activeRows = document.querySelectorAll(
            `.preview-edit-row[data-entity-type="${entityType}"]:not(.deleted)`
        );
        const countEl = document.getElementById('prevCount_' + entityType);
        if (countEl) countEl.textContent = activeRows.length;
    }

    // ── Update Total Labels stat in banner ───────────────────────
    const totalActive = document.querySelectorAll('.preview-edit-row:not(.deleted)').length;
    const totalEl = document.getElementById('prevStatTotal');
    if (totalEl) totalEl.textContent = totalActive;
}

/**
 * closePreview()
 * Hides the preview modal and cleans up event listeners.
 */
function closePreview() {
    document.getElementById('previewOverlay').classList.remove('open');
    document.removeEventListener('keydown', previewKeyHandler);
    pendingSaveStatus = null;
}

/**
 * closePreviewOnBackdrop(event)
 * Closes modal only when clicking the dark backdrop, not the modal itself.
 */
function closePreviewOnBackdrop(event) {
    if (event.target === document.getElementById('previewOverlay')) {
        closePreview();
    }
}

/**
 * previewKeyHandler(event)
 * Keyboard shortcuts while preview modal is open.
 */
/**
 * previewKeyHandler(event)
 * Keyboard shortcuts while preview modal is open.
 */
function previewKeyHandler(event) {
    if (event.key === 'Escape') {
        event.preventDefault();
        closePreview();
    }

    // 🐛 BUG FIX: Previously, pressing Enter while editing an input
    // forced the save status to 'in_progress' — even if the user opened
    // the preview via "✅ Mark Complete" (status = 'completed').
    // The user would see their status silently downgraded and think
    // the "completed" save didn't work. Drama! 🎭
    //
    // NOW: Enter in an input → save as Draft (in_progress) WITHOUT closing,
    // so the user can keep editing. This is a "quick save" convenience.
    // Enter outside an input → save with the INTENDED status and close.
    // Like hitting "Save" vs "Save & Close" in any good editor! 💾
    if (event.key === 'Enter') {
        const inInput = event.target.closest('.pf-input, .preview-edit-input, select');
        
        if (inInput) {
            event.preventDefault();
            // Quick-save as draft while staying in the preview
            confirmSave('in_progress', false); 
        } else {
            event.preventDefault();
            // Full save with the originally intended status, then close
            confirmSave(pendingSaveStatus, true); 
        }
    }
}

// =================================================================
// 💾 SAVE / LOAD
// =================================================================
/**
 * saveAnnotations(status)
 * Sends the final data payload to Python.
 */
function saveAnnotations(status = 'in_progress') {
    // 🛡️ Guard: ensure status is always a valid string
    // If somehow null/undefined leaks through (e.g. from pendingSaveStatus
    // being cleared by closePreview), default to 'in_progress'.
    if (!status) status = 'in_progress';

    const payload = {
        doc_id: DOC_ID,
        status: status,
        function: document.getElementById('functionSelect').value,
        industry: document.getElementById('industrySelect').value,
        hard_skills: currentHardSkills,
        soft_skills: currentSoftSkills,
        tags: currentTags,
        annotations: annotations,
        // 👤 Track who saved — stored in ner_documents.annotator
        annotator_name: getAnnotatorName() || ''
    };

    fetch('/api/save_annotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
    if (data.success) {
        if (status === 'completed') {
            showToast(
                '🏆 Marked Complete! Candidate is officially runway-ready! ✨',
                'success',
                6000
            );
            showCompletionCelebration();   // ← Big glamorous finale only for Complete
        } else {
            showToast(
                '💾 Draft Saved! Progress secured — come back anytime! ✨',
                'success',
                4000
            );
            showDraftSavedCelebration();   // ← Tiny cute party ONLY for drafts
        }
        updateStatusBadge(status);
    } else {
        showToast('❌ Save failed: ' + (data.error || 'Unknown error'), 'error', 6000);
    }
})
    .catch(err => {
        showToast('❌ Network error: ' + err, 'error', 6000);
    });
}

/**
 * updateStatusBadge(status)
 *
 * Updates the header status badge text + CSS class to reflect the
 * latest save status — live, no page reload needed.
 *
 * Status → badge mapping (mirrors Jinja2 logic in the template):
 *   completed   → badge-complete  ✅ completed
 *   in_progress → badge-progress  🔵 in_progress
 *   anything else → badge-pending ⏳ <status>
 *
 * Drama analogy: It's like the stage manager flipping the sign from
 * "REHEARSAL" to "PERFORMANCE" the moment the show goes live! 🎭✨
 *
 * @param {string} status - 'completed' | 'in_progress' | 'pending'
 */
function updateStatusBadge(status) {
    const badge = document.getElementById('statusBadge');
    if (!badge) return;  // Guard — badge must exist in DOM

    // Remove all existing badge-* colour classes before adding the new one
    badge.classList.remove('badge-complete', 'badge-progress', 'badge-pending');

    if (status === 'completed') {
        badge.classList.add('badge-complete');
        badge.textContent = 'completed';
    } else if (status === 'in_progress') {
        badge.classList.add('badge-progress');
        badge.textContent = 'in_progress';
    } else {
        badge.classList.add('badge-pending');
        badge.textContent = status || 'pending';
    }
}



