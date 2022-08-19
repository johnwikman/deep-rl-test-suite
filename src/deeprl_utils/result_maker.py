"""
Creates an HTML-file containing information about the results of the
experiment that was run.

Added by @dansah
"""

from datetime import datetime

def add_architecture(html_str, arch_name, diagrams_and_tables):
    """
    Adds a header for an architecture and image tags referring to the
    result diagrams provided.
    html_str (string): The HTML-string to append the result to.
    arch_name (string): Name of the architecture.
    diagrams (dict): { "diagrams": { "diagram_name1": "diagram_filepath1", ... }, "tables": { "table1": data1, ... } }
    """
    additional_str = f"<h3>{arch_name}</h3>"
    additional_str += f"<h3>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>"

    BORDER_STYLE = "1px solid black"

    diagrams = diagrams_and_tables.get("diagrams", {})
    for diagram_name, diagram_path in diagrams.items():
        additional_str += f"<img src=\"{diagram_path}\" alt=\"{diagram_name}\">"

    tables = diagrams_and_tables.get("tables", {})
    for table_name, table in tables.items():
        additional_str += f"<h4>{table_name}</h4><table style=\"border: {BORDER_STYLE}; border-spacing: 2px 2px;\">"
        for i in range(len(table)):
            additional_str += "<tr>"
            for j in range(len(table[i])):
                e = table[i][j]
                if i == 0:
                    additional_str += f"<th style=\"border: {BORDER_STYLE}; padding: 4px;\">{e}</th>"
                elif isinstance(e, float):
                    additional_str += f"<td style=\"border: {BORDER_STYLE}; padding: 4px;\">{e:%.3f}</td>"
                else:
                    additional_str += f"<td style=\"border: {BORDER_STYLE}; padding: 4px;\">{e}</td>"
            additional_str += "</tr>"
        additional_str += "</table>"

    return html_str + additional_str


def add_environment(html_str, env_name, architecture_data):
    """
    Adds a header for an environment. Calls add_architecture for
    every architecture used with the enivronment.
    html_str (string): The HTML-string to append the result to.
    env_name (string): The name of the environment.
    architecture_diagrams (dict): 
        { "arch_name1": { "diagrams": { "name1": "filepath1", ... }, "tables": { "name1": data1, ... } }, ... }
    """
    additional_str = """<h2>%s</h2>""" % env_name
    for arch_name in architecture_data:
        additional_str = add_architecture(additional_str, arch_name, architecture_data[arch_name])

    return html_str + additional_str


def make_html(output_file_str, environment_architecture_data):
    """
    output_file_str (string): Full filepath to the resulting output file.
    environment_architecture_data (dict): Should have the following format:
        {
            "environment_name1": {
                "arch_name1": {
                    "diagrams": {
                        "name1": "filepath1",
                        ...
                    },
                    "tables": {
                        "name1": data1,
                        ...
                    }
                },
                ...
            },
            ...
        }
    """
    # Initialize
    file = open(output_file_str, 'w')
    html_str = "<html><head></head><body>"

    # Add content
    for env_name in environment_architecture_data:
        html_str = add_environment(html_str, env_name, environment_architecture_data[env_name])

    # Finish up
    html_str += "</body></html>"
    file.write(html_str)
    file.close()
