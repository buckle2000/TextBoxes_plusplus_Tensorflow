if __name__ == "__main__":
    import sys
    try:
        export_dir = sys.argv[1]
    except IndexError:
        print(f"Usage: {sys.argv[0]} DEST")
        exit(1)
    from demo import export_model
    export_model(export_dir)
