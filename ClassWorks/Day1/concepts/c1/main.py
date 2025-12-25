
def calculate(p_des , p_post , p_post_no , p_no):
    denom = (p_post * p_des) + (p_post_no * p_no)
    if denom == 0:
        print("Invalid probability values.")
        return None
    
    num = p_post * p_des
    prob = num / denom
    
    print(f"Probability of having disease given a positive test: {prob:.4f}")

print("Scenario 1: False positive rate = 5%")
calculate(0.001 , 0.95 , 0.05 , 0.999)
print()

print("Scenario 2: False positive rate = 10%")
calculate(0.001 , 0.95 , 0.10 , 0.999)