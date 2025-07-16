from setuptools import find_packages, setup


setup(
    name="mcq_generator",
    version="0.0.1",
    author="Muhammad Owais",
    author_email="owaisiqbal2021@gmail.com",
    install_requires=["openai", "langchain", "streamlit", "python-dotenv", "PyPDF2", "langchain-community"],
    packages=find_packages()
)