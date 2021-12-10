# Docker file for the coffee ratings prediction project
# Group 3: Michelle, Arlin, Kristen, Berkay

# Use of rocker/tidyverse as the base image
FROM rocker/tidyverse@sha256:d0cd11790cc01deeb4b492fb1d4a4e0d5aa267b595fe686721cfc7c5e5e8a684

# install R packages
RUN apt-get update -qq && apt-get -y --no-install-recommends install \
	&& install2.r --error \
	--deps TRUE \
	tidyverse

# Fix warning message
# RUN apt-get install -y --no-install-recommends libxt6
RUN apt-get install libxt6

# Install R packages using install.packages
RUN Rscript -e "install.packages('kableExtra')"
RUN Rscript -e "install.packages('knitr')"
RUN Rscript -e "install.packages('rmarkdown')"

# Set up miniconda and environment path for miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH=/root/miniconda3/bin:${PATH} 


# Install Python 3 packages
RUN conda install --quiet --yes \
	docopt=0.6.2 \
	pandas=1.3.3	\
	scikit-learn=1.0 \
	requests=2.24.0 \
	seaborn=0.11.2 \
	numpy=1.21.2 \
