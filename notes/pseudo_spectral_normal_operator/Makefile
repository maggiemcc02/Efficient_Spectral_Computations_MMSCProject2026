MAIN = folding
SRC  = $(MAIN).tex references.bib

.PHONY: all clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(SRC)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc \
	      $(MAIN).bbl $(MAIN).blg

distclean: clean
	rm -f $(MAIN).pdf
