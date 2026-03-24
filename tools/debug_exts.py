from ml_runner.core.extensions.extension_manager import ExtensionManager

def main():
    ext_manager = ExtensionManager()
    classes = ext_manager.get_config_classes()
    print('Extension config class names:', list(classes.keys()))

if __name__ == '__main__':
    main()
