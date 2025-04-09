DOCUMENT = main
TEX_FILES = $(DOCUMENT).tex abstract.tex $(wildcard sections/*.tex)
BIB_FILE = references.bib
OUTPUT_DIR = build

LATEX = pdflatex
BIBTEX = bibtex
LATEX_OPTS = -interaction=nonstopmode -output-directory=$(OUTPUT_DIR)
VIEWER = xdg-open

all: $(OUTPUT_DIR)/$(DOCUMENT).pdf

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OUTPUT_DIR)/$(DOCUMENT).pdf: $(TEX_FILES) $(BIB_FILE) | $(OUTPUT_DIR)
	# Copy references.bib to build directory to ensure BibTeX can find it
	cp $(BIB_FILE) $(OUTPUT_DIR)/
	$(LATEX) $(LATEX_OPTS) $(DOCUMENT)
	cd $(OUTPUT_DIR) && $(BIBTEX) $(DOCUMENT)
	$(LATEX) $(LATEX_OPTS) $(DOCUMENT)
	$(LATEX) $(LATEX_OPTS) $(DOCUMENT)
	@echo "PDF compilation complete: $(OUTPUT_DIR)/$(DOCUMENT).pdf"

quick: $(TEX_FILES) | $(OUTPUT_DIR)
	$(LATEX) $(LATEX_OPTS) $(DOCUMENT)
	@echo "Quick build complete: $(OUTPUT_DIR)/$(DOCUMENT).pdf"

clean:
	rm -f $(OUTPUT_DIR)/*.aux $(OUTPUT_DIR)/*.log $(OUTPUT_DIR)/*.bbl \
	      $(OUTPUT_DIR)/*.blg $(OUTPUT_DIR)/*.out $(OUTPUT_DIR)/*.toc \
	      $(OUTPUT_DIR)/*.lof $(OUTPUT_DIR)/*.lot $(OUTPUT_DIR)/*.fls \
	      $(OUTPUT_DIR)/*.fdb_latexmk $(OUTPUT_DIR)/*.synctex.gz

cleanall: clean
	rm -f $(OUTPUT_DIR)/$(DOCUMENT).pdf

cleanroot:
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz *.toc *.lof *.lot

view: $(OUTPUT_DIR)/$(DOCUMENT).pdf
	$(VIEWER) $< &

link: $(OUTPUT_DIR)/$(DOCUMENT).pdf
	ln -sf $(OUTPUT_DIR)/$(DOCUMENT).pdf ./$(DOCUMENT).pdf
	@echo "Created symbolic link to the PDF in the root directory"

watch:
	@echo "Watching for changes in .tex files..."
	@while true; do \
		inotifywait -e modify -r $(TEX_FILES) $(BIB_FILE); \
		make all; \
	done

full: cleanroot all link
	@echo "Full build completed and linked to root directory"

debug: | $(OUTPUT_DIR)
	@echo "--- Debugging LaTeX setup ---"
	@echo "Checking if references.bib exists in build directory..."
	@if [ -f "$(OUTPUT_DIR)/$(BIB_FILE)" ]; then \
		echo "✓ References file found in build directory"; \
	else \
		echo "✗ References file NOT found in build directory"; \
		cp $(BIB_FILE) $(OUTPUT_DIR)/; \
		echo "  → Copied references.bib to build directory"; \
	fi
	@echo "Running single LaTeX pass to generate aux file..."
	$(LATEX) $(LATEX_OPTS) $(DOCUMENT)
	@echo "Running BibTeX with verbose output..."
	cd $(OUTPUT_DIR) && $(BIBTEX) -terse $(DOCUMENT)
	@echo "--- Debug complete ---"
	@echo "If there were BibTeX errors, check that:"
	@echo "1. Citation keys in the text match those in references.bib"
	@echo "2. references.bib contains valid BibTeX entries"
	@echo "3. The apalike style is properly installed"

.PHONY: all clean cleanall view cleanroot link watch full debug