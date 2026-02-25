/**
 * Google Apps Script to generate a Google Slides presentation
 * from a Markdown file with sections like:
 *   # Slide 1 – Title
 *   - Bullet 1
 *   - Bullet 2
 *
 * How to use:
 * 1) Upload your `presentation_slides.md` to Google Drive.
 * 2) Create a new Apps Script project (script.google.com), or open an existing one.
 * 3) Add this file's contents to the project.
 * 4) Choose one content source (priority order):
 *    a) Set Script Property `PRESENTATION_MD_URL` to a raw URL of the markdown
 *       (e.g., https://raw.githubusercontent.com/<org>/<repo>/main/presentation_slides.md)
 *    b) Set Script Property `PRESENTATION_MD_INLINE_CONTENT` to the full markdown text
 *       (paste content; suitable for smaller files)
 *    c) Set Script Property `PRESENTATION_MD_FILE_ID` to the Drive file ID
 *    d) Ensure a file named `presentation_slides.md` exists in Drive (default search)
 * 5) Run `generateSlidesFromMarkdown()` and authorize.
 * 6) Check the execution log for the created presentation URL.
 *
 * Updating an existing Slides deck (optional):
 * - Set Script Property `PRESENTATION_ID` to the target Slides file ID
 *   (the long string after /d/ in the URL), then run
 *   `generateSlidesIntoExisting()`.
 * - To replace all existing slides instead of appending, set
 *   Script Property `PRESENTATION_REPLACE_ALL` to `true`.
 */

/**
 * Entry point: Generates a Slides presentation from the Markdown file.
 */
function generateSlidesFromMarkdown() {
  var markdown = getMarkdown_();
  var slidesData = parseSlidesFromMarkdown_(markdown);
  if (!slidesData.length) {
    throw new Error('No slides parsed. Ensure your markdown uses "# Slide N – Title" and "- Bullet" lines.');
  }

  var firstTitle = slidesData[0] && slidesData[0].title ? slidesData[0].title : 'Generated Slides';
  var dateStr = Utilities.formatDate(new Date(), Session.getScriptTimeZone() || 'Etc/UTC', 'yyyy-MM-dd');
  var presTitle = firstTitle.replace(/\s*Overview\s*$/i, '').trim();
  if (!presTitle) presTitle = 'Generated Slides';
  presTitle += ' (Generated ' + dateStr + ')';

  var presentation = SlidesApp.create(presTitle);

  // Remove the default first slide for a clean start.
  var existing = presentation.getSlides();
  if (existing && existing.length) {
    existing[0].remove();
  }

  // Render slides
  renderSlidesIntoPresentation_(presentation, slidesData);

  Logger.log('Created presentation: ' + presentation.getUrl());
}

/**
 * Entry point: Appends or replaces slides in an existing presentation.
 * Requires Script Property `PRESENTATION_ID`.
 * Optional: set `PRESENTATION_REPLACE_ALL` to `true` to replace all existing slides.
 */
function generateSlidesIntoExisting() {
  var markdown = getMarkdown_();
  var slidesData = parseSlidesFromMarkdown_(markdown);
  if (!slidesData.length) {
    throw new Error('No slides parsed. Ensure your markdown uses "# Slide N – Title" and "- Bullet" lines.');
  }

  var props = PropertiesService.getScriptProperties();
  var presId = (props.getProperty('PRESENTATION_ID') || '').trim();
  if (!presId) {
    throw new Error('Set Script Property `PRESENTATION_ID` to the Slides file ID (from the /d/<ID>/ URL).');
  }

  var presentation = SlidesApp.openById(presId);

  var replaceAll = String(props.getProperty('PRESENTATION_REPLACE_ALL') || '').toLowerCase() === 'true';
  if (replaceAll) {
    // Snapshot original slides, render new ones, then remove originals to avoid "need at least one slide" constraints.
    var originalSlides = presentation.getSlides().slice();
    renderSlidesIntoPresentation_(presentation, slidesData);
    originalSlides.forEach(function(s) { try { s.remove(); } catch (e) {} });
  } else {
    renderSlidesIntoPresentation_(presentation, slidesData);
  }

  Logger.log('Updated presentation: ' + presentation.getUrl());
}

/**
 * Attempts to get the markdown content from Drive.
 * Priority: Script Property `PRESENTATION_MD_FILE_ID` > file named `presentation_slides.md`.
 */
function getMarkdown_() {
  var props = PropertiesService.getScriptProperties();
  var url = (props.getProperty('PRESENTATION_MD_URL') || '').trim();
  var inline = props.getProperty('PRESENTATION_MD_INLINE_CONTENT');
  var fileId = props.getProperty('PRESENTATION_MD_FILE_ID');
  var fileName = props.getProperty('PRESENTATION_MD_FILE_NAME') || 'presentation_slides.md';

  // Priority 1: URL fetch
  if (url) {
    try {
      var resp = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
      var code = resp.getResponseCode();
      if (code >= 200 && code < 300) {
        var body = resp.getContentText();
        if (body && body.trim()) return body;
        throw new Error('Fetched URL is empty.');
      }
      throw new Error('Failed to fetch URL. HTTP ' + code + ' from ' + url);
    } catch (e) {
      throw new Error('Error fetching `PRESENTATION_MD_URL`: ' + e);
    }
  }

  // Priority 2: Inline content
  if (inline && inline.trim()) {
    return inline;
  }

  // Priority 3: Drive file by ID
  var file;
  if (fileId && fileId.trim()) {
    try {
      file = DriveApp.getFileById(fileId.trim());
    } catch (e) {
      throw new Error('Failed to open file by ID. Set a valid `PRESENTATION_MD_FILE_ID`.\n' + e);
    }
  } else {
    // Priority 4: First file by name in Drive
    var files = DriveApp.getFilesByName(fileName);
    if (!files.hasNext()) {
      throw new Error('Markdown not found. Configure one of: \n' +
                      ' - Script Property `PRESENTATION_MD_URL` (raw markdown URL)\n' +
                      ' - Script Property `PRESENTATION_MD_INLINE_CONTENT` (paste content)\n' +
                      ' - Script Property `PRESENTATION_MD_FILE_ID` (Drive file ID)\n' +
                      ' - Upload a file named `' + fileName + '` to your Drive');
    }
    file = files.next();
  }

  var blob = file.getBlob();
  var content = blob.getDataAsString();
  if (!content || !content.trim()) {
    throw new Error('Markdown file is empty.');
  }
  return content;
}

/**
 * Parses slides from markdown.
 * Expected format:
 *   # Slide 1 – Title text
 *   - Bullet line 1
 *   - Bullet line 2
 */
function parseSlidesFromMarkdown_(markdown) {
  var text = (markdown || '').replace(/\r\n/g, '\n');
  var lines = text.split('\n');

  var slides = [];
  var current = null;
  var headerRegex = /^#\s*Slide\s*\d+\s*[–—-]\s*(.+)$/i; // en dash, em dash, or hyphen
  var bulletRegex = /^\s*-\s+(.*)$/;

  lines.forEach(function(line) {
    var headerMatch = line.match(headerRegex);
    if (headerMatch) {
      if (current) slides.push(current);
      current = { title: sanitizeText_(headerMatch[1]), bullets: [] };
      return;
    }

    if (!current) {
      // Ignore any preamble text before first header.
      return;
    }

    var bulletMatch = line.match(bulletRegex);
    if (bulletMatch) {
      var bulletText = sanitizeText_(bulletMatch[1]);
      if (bulletText) current.bullets.push(bulletText);
      return;
    }

    // If non-empty, non-bullet lines appear under a slide, treat them as bullets.
    var trimmed = line.trim();
    if (trimmed) {
      current.bullets.push(sanitizeText_(trimmed));
    }
  });

  if (current) slides.push(current);
  return slides;
}

/**
 * Finds a TITLE_AND_BODY layout; falls back to the first layout if unavailable.
 */
function getTitleAndBodyLayout_(presentation) {
  var layouts = presentation.getLayouts();
  for (var i = 0; i < layouts.length; i++) {
    if (layouts[i].getLayoutName && layouts[i].getLayoutName() === 'TITLE_AND_BODY') {
      return layouts[i];
    }
  }
  // Fallback: return first layout if specific one not found.
  return layouts[0];
}

/**
 * Basic text sanitization for slide content.
 */
function sanitizeText_(s) {
  if (!s) return '';
  // Trim and normalize spaces; keep punctuation and symbols (e.g., en dashes, arrows).
  var t = String(s).replace(/\s+/g, ' ').trim();
  return t;
}

/**
 * Renders the slides into the provided presentation using a Title & Body layout.
 */
function renderSlidesIntoPresentation_(presentation, slidesData) {
  var layout = getTitleAndBodyLayout_(presentation);
  var bulletPreset = SlidesApp.ListPreset.BULLET_DISC_CIRCLE_SQUARE;
  var titleFontSize = 30; // points
  var bodyFontSize = 18;  // points

  slidesData.forEach(function(sd) {
    var slide = presentation.appendSlide(layout);

    // Title
    var titlePlaceholder = slide.getPlaceholder(SlidesApp.PlaceholderType.TITLE);
    if (titlePlaceholder) {
      var titleShape = titlePlaceholder.asShape();
      var titleText = titleShape.getText();
      titleText.setText(sd.title || '');
      titleText.getTextStyle().setBold(true).setFontSize(titleFontSize);
    }

    // Body bullets
    var bodyPlaceholder = slide.getPlaceholder(SlidesApp.PlaceholderType.BODY);
    if (bodyPlaceholder) {
      var bodyShape = bodyPlaceholder.asShape();
      var textRange = bodyShape.getText();
      textRange.setText('');

      if (sd.bullets && sd.bullets.length) {
        sd.bullets.forEach(function(line, idx) {
          if (idx > 0) textRange.appendText('\n');
          textRange.appendText(line);
        });
        var style = bodyShape.getText().getTextStyle();
        style.setFontSize(bodyFontSize);
        bodyShape.getText().getListStyle().applyListPreset(bulletPreset);
      }
    }
  });
}
