FROM jupyter/minimal-notebook@sha256:f424aa7321d0dd5543ac86c3efa8ec583efd8c1771266ced1b3449487e0adb63

# Install Python 3 packages
RUN conda install --quiet --yes \
	'altair=4.1.*' \
	'docopt=0.6.2' \
	'pandas=1.3.3'	\
	'scikit-learn=1.0' \
	'requests=2.24.0' \
	'altair=4.1.0' \
	'altair_saver=0.5.*' \
	'seaborn=0.11.2' \
	'numpy=1.21.2'

# Install R packages
RUN conda install --quiet --yes \
	'r-base=4.0.5' \
	'r-knitr=1.36*' \
	'r-rmarkdown=2.11*' \
	'r-kableExtra=1.3.4 
	
	