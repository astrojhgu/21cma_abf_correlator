targets:ms.pdf


ms.pdf:ms.tex 
	xelatex ms;bibtex ms;xelatex $<;xelatex $<

clean:
	rm -f ms.pdf *.aux *.log *~ *.bbl *.blg *.out *.fls *-converted-to.pdf *synctex.gz *.dot *.ps *.dvi *.fdb_latexmk fig_dependences.pdf fig_flow_chart.pdf
