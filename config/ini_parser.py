import configparser

from registries import ConfigRegistry

@ConfigRegistry.register(".ini")
def parse_ini(path):
    parser = configparser.ConfigParser()
    parser.read(path)

    return {section: dict(parser[section]) for section in parser.sections()}
