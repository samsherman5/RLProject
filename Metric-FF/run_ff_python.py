import subprocess

def run_ff_metric(domain_file, problem_file, plan_output):
    # Run FF-Metric with the domain and problem files
    planner_command = [
        "./ff",  # Path to the FF-Metric executable (ensure it is correct)
        "-o", domain_file,
        "-f", problem_file
    ]

    # Execute the planner and capture the output
    result = subprocess.run(planner_command, capture_output=True, text=True)

    # Save the output plan to a file
    with open(plan_output, "w") as plan_file:
        plan_file.write(result.stdout)

    return result.stdout

# Example usage
domain_file = "domain.pddl"
problem_file = "problem.pddl"
plan_output = "ff_plan.txt"

plan = run_ff_metric(domain_file, problem_file, plan_output)
print("FF Metric Planner Output:", plan)

def parse_ff_plan(plan_output):
    plan = []
    parsing = False  # Flag to start parsing after "found legal plan"
    plan_started = False  # Flag to determine if we have reached the step section

    with open(plan_output, "r") as plan_file:
        lines = plan_file.readlines()

    step_flag = 0
    for line in lines:
        line = line.strip()

        # Check if we are in the steps section or not
        if "ff: found legal plan as follows" in line:
            parsing = True
            plan_started = True
            continue

        if parsing:
            if line.startswith("step"):
                # Extract action part after the colon
                action = line.split(":", 1)[1].strip()
                plan.append(action)
                step_flag = 1
            elif "time spent:" in line:
                # Stop parsing when "time spent:" section starts
                break
            elif step_flag == 1 and not line == "":
                print("line: ",line)
                action = line.split(":")[1][1:]
                print("action:", action)
                plan.append(action)                

    return plan

# Example usage
parsed_plan = parse_ff_plan("ff_plan.txt")
print("Parsed Plan:", parsed_plan)
