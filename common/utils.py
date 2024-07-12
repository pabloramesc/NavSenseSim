
def progress_bar(part,total,prompt):
    cont = int(part/total*50)
    print(" "+str(part)+"/"+str(total),prompt," ["+cont*"#"+(50-cont)*"-"+"]",cont*2,"%", end='\r')
    if cont == 50: print()


    