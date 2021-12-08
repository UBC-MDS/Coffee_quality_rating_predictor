# Docker file for the coffee ratings prediction project

# use rocker/tidyverse as the base image
FROM rocker/tidyverse@sha256:d0cd11790cc01deeb4b492fb1d4a4e0d5aa267b595fe686721cfc7c5e5e8a684

# install R packages
RUN apt-get update -qq && apt-get -y --no-install-recommends install \
	&& install2.r --error \
	--deps TRUE \
	knitr \
	rmarkdown
# cowsay \
# here \
# feather \
# ggridges \
# ggthemes \
# e1071 \
# caret \


# install the kableExtra package using install.packages
RUN Rscript -e "install.packages('kableExtra')"

# install the anaconda distribution of python
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
	/bin/bash ~/anaconda.sh -b -p /opt/conda && \
	rm ~/anaconda.sh && \
	ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
	echo "conda activate base" >> ~/.bashrc && \
	find /opt/conda/ -follow -type f -name '*.a' -delete && \
	find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
	/opt/conda/bin/conda clean -afy && \
	/opt/conda/bin/conda update -n base -c defaults conda

# put anaconda python in path
ENV PATH="/opt/conda/bin:${PATH}"


# Install Python 3 packages
RUN conda install --quiet --yes \
	altair=4.1.* \
	docopt=0.6.2 \
	pandas=1.3.3	\
	scikit-learn=1.0 \
	requests=2.24.0 \
	seaborn=0.11.2 \
	numpy=1.21.2 \