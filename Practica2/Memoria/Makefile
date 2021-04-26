# Parametros del programa
# FILE_NAME puede llevar espacios sin problemas, sin necesidad de comillas dobles
FILE_NAME=Memoria

all:
	# Tenemos que ejecutar esta secuencia para que se cargue correctamente la bibliografia
	pdflatex "${FILE_NAME}.tex" -interaction=nonstopmode
	# TODO -- descomentar esto para que la biliografia se genere correctamente
	#bibtex "${FILE_NAME}"
	pdflatex "${FILE_NAME}.tex" -interaction=nonstopmode
	pdflatex "${FILE_NAME}.tex" -interaction=nonstopmode
clean:
	@echo "Borrando archivos auxiliares"
	@rm -rf *.aux *.log *.toc *.bbl *.blg
