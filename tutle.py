while True:
  string = input("Enter a string of characters: ")
  vowels = 0
  remaining_chars = 0
  times_a=0
  times_e=0
  times_i=0
  times_o=0
  times_u=0
  for char in string:
    match char.lower():
      case 'a':
        times_a += 1
      case 'e':
        times_e += 1  
      case 'i':
        times_i += 1
      case 'o':
        times_o += 1
      case 'u':
        times_u += 1      
      case _:
        remaining_chars += 1
  vowels=times_u+times_a+times_e+times_i+times_o   
  print("Number of each vowel in the string:")

  print(f"a:{times_a}")
  print(f"e:{times_e}")
  print(f"i:{times_i}")
  print(f"o:{times_o}")
  print(f"u:{times_u}")
  print(f"Number of non-vowel characters: {remaining_chars}")
  print("Would you like to enter another sting?")
  choice = input("Enter yes or no: ")
  if choice.lower() != 'yes':
    break

