import os

def main():
    models_dir = 'models'
    output_file = 'all_architectures.txt'
    
    # Get all neon*.py files and sort them numerically
    files = [f for f in os.listdir(models_dir) if f.startswith('neon') and f.endswith('.py')]
    files.sort()
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in files:
            filepath = os.path.join(models_dir, filename)
            name = filename.replace('.py', '')
            outfile.write(f"--- {name} ---\n")
            try:
                with open(filepath, 'r', encoding='utf-8') as infile:
                    # Read the first 2000 characters (covers headers and class definitions)
                    content = infile.read(2000)
                    outfile.write(content)
            except Exception as e:
                outfile.write(f"Error reading file: {e}\n")
            outfile.write("\n\n")
    
    print(f"Successfully aggregated {len(files)} models into {output_file}")

if __name__ == "__main__":
    main()
