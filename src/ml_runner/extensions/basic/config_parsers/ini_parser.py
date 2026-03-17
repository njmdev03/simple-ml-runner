import configparser

from ml_runner.core.registries import ConfigParser

@ConfigParser(".ini")
def parse_ini(path):
    parser = configparser.ConfigParser()
    parser.read(path)

    return {section: dict(parser[section]) for section in parser.sections()}
