TEXFILES = $(wildcard *.tex)
PDFFILES = $(TEXFILES:.tex=.pdf)
PNGFILES = $(PDFFILES:.pdf=.png)

all: pdf clean

pdf: $(PDFFILES)

%.pdf: %.tex
	@echo $(TEXFILES)
	@rubber --pdf $<
	@if [ -d publish ];then mv *.pdf publish; else mkdir publish; mv *.pdf publish/;fi

clean:
	@rubber --clean $(TEXFILES:.tex=)

distclean: clean
	@rubber --clean --pdf $(TEXFILES:.tex=)
	@rm -rf publish

x:
	@evince publish/presentacion.pdf
