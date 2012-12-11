TEXFILES = $(wildcard *.tex)
PDFFILES = $(TEXFILES:.tex=.pdf)
PNGFILES = $(PDFFILES:.pdf=.png)

all: pdf clean

pdf: $(PDFFILES)

%.pdf: %.tex
	@echo $(TEXFILES)
	@rubber --pdf $<
	@if [ -d bin ];then mv *.pdf bin; else mkdir bin; mv *.pdf bin/;fi

clean:
	@rubber --clean $(TEXFILES:.tex=)
	@rm -f *.out *.pdf

distclean: clean
	@rubber --clean --pdf $(TEXFILES:.tex=)
	@rm -rf bin

x:
	@evince bin/$(PDFFILES)
