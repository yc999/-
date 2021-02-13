package main

import "fmt"

func solvedata() {
	return
}

func main() {
	num, _ := Hammingdistance("000782", "000725")
	fmt.Println(num)
}

func Hammingdistance(sourceStr string, destinateStr string) (result int, resultplace [2]int) {
	lensource := len(sourceStr)
	lendestinate := len(destinateStr)

	var differntcount int = 0
	var place [2]int
	if lendestinate != lensource {
		return 0, place
	}
	for i := 0; i < lendestinate; i++ {
		if sourceStr[i] != destinateStr[i] {
			if differntcount == 2 {
				return 0, place
			}
			place[differntcount] = i
			differntcount++
		}
	}
	return differntcount, place
}
