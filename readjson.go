package main

//{"timestamp":"1600993027","name":"0.205-240-81.adsl-dyn.isp.belgacom.be","type":"a",
//"value":"81.240.205.0"}
import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
)

//DNS json文件中DNS日志格式
type DNS struct {
	Timestamp string `json:"timestamp"`
	Name      string `json:"name"`
	Dnstype   string `json:"type"`
	Value     string `json:"value"`
}

//域名信息
type othernameinfo struct {
	Value    map[string][]int64 //其他域名 : []timestamp(10秒)
	sumquery int                //总次数
}

//每10秒的总访问量
type sumquerySecond struct {
	// Timestamp int64
	sumquery  int64          //请求总数
	fqdntimes map[string]int //域名 ： 请求数
}

// Timestamp（10秒间隔） ： 总访问量
var mapSumquery map[int64]*sumquerySecond

// ip ：二级域名 : 域名信息  如果有国家顶级域名后缀则算入二级域名
var secondNameMap map[string]map[string]*othernameinfo

// ip ：二级域名 ： 域名信息  短域名只有二级域名和顶级域名
var secondNameMapShort map[string]map[string]*othernameinfo

//
var cnamemap map[string]map[string][]int

//保存短域名集合  二级域名 : 顶级域名
var shortnameMap map[string]string

// 相似哈希表 保存所有相似的名称 timestamp ： {[name], []}
var similarName map[string][]string

//cdn字典
var cdn2nameMap map[string][]string
var name2cdnMap map[string][]string

// 国家顶级域名后缀 开头字母 ：后缀字符串数组
var counter = struct {
	sync.RWMutex
	cctld map[string][]string
}{cctld: make(map[string][]string)}

//读取数据行数
var looptimes int = 130000000

func main() {
	path := "D:/dnswork/sharevm/2020-09-25-1600992674-fdns_a.json/2020-09-25-1600992674-fdns_a.json"
	cctldpath := "D:/dnswork/sharevm/suffix"
	pkg := DNS{}
	initglobal()
	// subNameMap := make(map[string]map[string]*othernameinfo)
	// cctld = make(map[string][]string)
	getcctld(cctldpath)

	file, err := os.Open(path)
	checkError(err)
	defer file.Close()
	readerT := bufio.NewReader(file)
	countT := 0
	for true {
		strT, errT := readerT.ReadString('\n')
		if errT != nil {
			if errT == io.EOF {
				strT = strings.TrimRight(strT, "\r\n")
				countT++
			}
			break
		}
		countT++
		err = json.Unmarshal([]byte(strT), &pkg) //解析json
		checkError(err)
		solvedata(&pkg)
		if countT >= looptimes {
			// for k, v := range mapSumquery {
			// 	if v.sumquery > 5 {
			// 		fmt.Println(k, v)
			// 	}
			// }
			break
		}
	}
	fmt.Print(sortnameSlice)

	// fmt.Println(cctld)
	// writepath := "D:/dnswork/sharevm/secondname"
	// f,err := os.Create(writepath)
	// defer f.Close()
	// fmt.Println(secondNameMap)
	// fmt.Println(secondNameMapShort)

	// for k, v := range secondNameMap {
	// 	fmt.Println(k)
	// 	for kk, vv := range v {
	// 		fmt.Println(kk)
	// 		for _, vvv := range vv {
	// 			fmt.Println(vvv)
	// 		}
	// 	}
	// }
}

//用于统计前访问次数最多的二级域名
type sortname struct {
	name  string //二级域名
	count int    //访问次数
}

var sortnum int = 1000
var sortnameSlice [1000]sortname

var maxquerytimes int
var minquerytimes int = 0

//记录总共
var sortnamecount int = 0

// 初始化参数
func initglobal() {
	shortnameMap = make(map[string]string)
	similarName = make(map[string][]string)
	mapSumquery = make(map[int64]*sumquerySecond)
	secondNameMap = make(map[string]map[string]*othernameinfo)
	secondNameMapShort = make(map[string]map[string]*othernameinfo)
	cdn2nameMap = make(map[string][]string)
}

//提取所有国家顶级域名
func getcctld(path string) {
	f, err := os.Open(path)
	checkError(err)
	defer f.Close()
	buf := bufio.NewReader(f)
	counter.RLock()
	for {
		line, err := buf.ReadString('\n')
		parts := strings.Split(line, " :")
		counter.cctld[parts[0][0:1]] = append(counter.cctld[parts[0][0:1]], parts[0])
		if err != nil {
			if err == io.EOF {
				break
			}
			panic(err)
		}
	}
	counter.RUnlock()
}

//检测错误
func checkError(err error) {
	if err != nil {
		panic(err)
	}
}

func solvedata(data *DNS) {
	dnstime, err := strconv.ParseInt(data.Timestamp, 10, 64)
	dnstime = dnstime / 100000000
	if err != nil {
		fmt.Println("wrong message:")
		fmt.Println(data)
		return
	}
	// checkError(err)
	// secondname, othername, nametype := parsename(data)
	// secondname, othername, nametype := parsename(data)
	// if nametype == 1 {
	// findsimlarname(secondname, othername, data.Name)
	// }

	// fmt.Println(data.Dnstype)
	if data.Dnstype == "a" || data.Dnstype == "aaaa" {
		secondname, _, nametype := parsename(data)
		// findsimlarname(secondname, othername, data.Name)
		insert2sumquery(dnstime, data, secondname, nametype)
		// fmt.Println(data)
	} else if data.Dnstype == "cname" {
		// cnameanalyze(data)
	} else {
		fmt.Println(data)
	}
	//insert2sumquery(dnstime, data, secondname, nametype)
	// if nametype == 1 {
	// 	collectsecondname(dnstime, data, secondname, othername, secondNameMapShort)
	// } else {
	// 	collectsecondname(dnstime, data, secondname, othername, secondNameMap)
	// }
	return
}

//分析cname记录
func cnameanalyze(data *DNS) {
	parts := strings.Split(data.Name, ".")
	partslen := len(parts)
	cnameparts := strings.Split(data.Value, ".")
	cnamepartslen := len(cnameparts)
	cnamesecondname := strings.Join(cnameparts[cnamepartslen-2:cnamepartslen], ".")
	secondname := strings.Join(parts[partslen-2:partslen], ".")
	if strings.Compare(cnamesecondname, secondname) != 0 {
		// fmt.Println(secondname, cnamesecondname)
		cnamesecondname = data.Value
		_, have := cdn2nameMap[cnamesecondname]
		if have {
			if !findinslicestr(secondname, cdn2nameMap[cnamesecondname]) {
				cdn2nameMap[cnamesecondname] = append(cdn2nameMap[cnamesecondname], secondname)
			}
		} else {
			cdn2nameMap[cnamesecondname] = append(cdn2nameMap[cnamesecondname], secondname)
		}
	}
}

func findsimlarname(secondname string, othername string, dataname string) {
	_, ok := shortnameMap[secondname]
	if !ok {
		for targetname := range shortnameMap {
			result, _ := Hammingdistance(secondname, targetname)
			// if len(secondname) == len(targetname) {
			// 	fmt.Println(targetname, secondname, result)
			// }
			if result <= 3 { // 如果是短域名并且距离小于等于2 加入相似哈希表中
				_, have := similarName[targetname]
				if have {
					if !findinslicestr(dataname, similarName[targetname]) {
						// fmt.Println(secondname, targetname)
						similarName[targetname] = append(similarName[targetname], dataname)
						return
					}
				} else {
					similarName[targetname] = append(similarName[targetname], dataname)
					return
				}
			}
		}
		_, have := shortnameMap[secondname]
		if !have {
			shortnameMap[secondname] = othername
		}
	}
}

//找到出现最多的次数的域名，
func findfrequentname(name string, time int) {
	if time > minquerytimes {
		for i := 0; i < sortnamecount; i++ {
			if sortnameSlice[i].name == name { //防止域名重复
				sortnameSlice[i].count = time
				return
			}
		}
		if sortnamecount < sortnum { //数组未满
			sortnameSlice[sortnamecount].name = name
			sortnameSlice[sortnamecount].count = time
			sortnamecount++
		} else {
			for i := 0; i < sortnum; i++ { //数组满， 替换最小的域名
				if sortnameSlice[i].count == minquerytimes {
					sortnameSlice[i].count = time
					sortnameSlice[i].name = name
					break
				}
			}
			//重新确定 minquerytimes
			minquerytimes = sortnameSlice[0].count
			for i := 1; i < sortnum; i++ {
				if sortnameSlice[i].count < minquerytimes {
					minquerytimes = sortnameSlice[i].count
				}
			}
		}
	}
}

//将数据加入sumquery_per_second中
func insert2sumquery(dnstime int64, data *DNS, secondname string, nametype int) {
	historydata, have := mapSumquery[dnstime] //判断时间是否重复出现，重复则查询总数加1，
	if have {
		// historydata.sumquery++
		mapSumquery[dnstime].sumquery++
		ok := historydata.fqdntimes[secondname] //判断当前时间域名是否曾经出现
		if ok != 0 {                            //出现过
			mapSumquery[dnstime].fqdntimes[secondname]++
			findfrequentname(secondname, mapSumquery[dnstime].fqdntimes[secondname])
		} else { //未出现过
			mapSumquery[dnstime].fqdntimes[secondname] = 1
		}
	} else { //未出现
		mapSumquery[dnstime] = &sumquerySecond{sumquery: 1,
			fqdntimes: map[string]int{secondname: 1},
		}
	}
}

// 返回处理后的域名，只有二级域名则返回二级域名和顶级域名 nametype表示是短域名或长域名 1表示短域名
// 如 a.b.c.com  拆分为  c.com 和 a.b
// a.b.c.com.cn 拆分为 c.com.cn 和 a.b
// c.com 拆分为 c.com 和 空
func parsename(data *DNS) (secondname string, othername string, nametype int) {
	var returentype int = 0
	parts := strings.Split(data.Name, ".")
	partslen := len(parts)
	tldname := parts[partslen-1]
	var tmpSecondname, tmpOthername string
	if partslen >= 3 {
		if findinslicestr(tldname, counter.cctld[tldname[0:1]]) && partslen > 3 {
			tmpOthername = strings.Join(parts[0:partslen-3], ".")
			tmpSecondname = strings.Join(parts[partslen-3:partslen], ".")
		} else {
			tmpOthername = strings.Join(parts[0:partslen-2], ".")
			tmpSecondname = strings.Join(parts[partslen-2:partslen], ".")
		}
	} else {
		tmpSecondname = data.Name
		tmpOthername = ""
		returentype = 1
	}
	return tmpSecondname, tmpOthername, returentype
}

// Hammingdistance 返回两个字符串汉明距离小于2的情况，并返回字符不同处的位置，距离大于2立刻返回
func Hammingdistance(sourceStr string, destinateStr string) (result int, resultplace [2]int) {
	lensource := len(sourceStr)
	lendestinate := len(destinateStr)

	var differntcount int = 0
	var place [2]int
	if lendestinate != lensource {
		return 4, place
	}
	for i := 0; i < lendestinate; i++ {
		if sourceStr[i] != destinateStr[i] {
			if differntcount == 2 { //阈值为 2
				return 4, place
			}
			place[differntcount] = i
			differntcount++
		}
	}
	return differntcount, place
}

// Levenshtein 判断两个字符串的编辑距离
func Levenshtein(sourceStr string, destinateStr string) int {
	lensource := len(sourceStr)
	lendestinate := len(destinateStr)
	if lensource == 0 {
		return lendestinate
	} else if lendestinate == 0 {
		return lensource
	}
	var dif [][]int
	for x := 0; x < lensource+1; x++ {
		arr := make([]int, lendestinate+1)
		dif = append(dif, arr)
	}
	for i := 0; i < lensource; i++ {
		dif[i][0] = i
	}
	for i := 0; i < lensource; i++ {
		dif[0][i] = i
	}
	var temp int
	for i := 1; i <= lensource; i++ {
		for j := 1; j < lendestinate; j++ {
			if sourceStr[i-1] == destinateStr[j-1] {
				temp = 0
			} else {
				temp = 1
			}
			dif[i][j] = min(dif[i-1][j-1]+temp,
				dif[i][j-1]+1,
				dif[i-1][j]+1)
		}
	}
	return dif[lensource][lendestinate]
}

//拆分为二级域名和其他，保存在 secondNameMap 中
func collectsecondname(dnstime int64, data *DNS, tmpSecondname string,
	tmpOthername string, targetmap map[string]map[string]map[string]*othernameinfo) {
	if data.Dnstype != "a" && data.Dnstype != "cname" {
		fmt.Println(data)
	}
	//处理多维map
	_, exist0 := targetmap[data.Value]
	if exist0 {
		_, exist1 := targetmap[data.Value][tmpSecondname]
		if exist1 {
			_, exist2 := targetmap[data.Value][tmpSecondname][data.Dnstype]
			if exist2 {
				targetmap[data.Value][tmpSecondname][data.Dnstype].sumquery++
				// othername不存在, 添加到数组中
				if !findinsliceint64(dnstime, targetmap[data.Value][tmpSecondname][data.Dnstype].Value[tmpOthername]) {
					targetmap[data.Value][tmpSecondname][data.Dnstype].Value[tmpOthername] = append(
						targetmap[data.Value][tmpSecondname][data.Dnstype].Value[tmpOthername],
						dnstime)
				}
			} else { //Dnstype不存在, 新增一个
				targetmap[data.Value][tmpSecondname][data.Dnstype] = &othernameinfo{
					sumquery: 1,
					Value:    map[string][]int64{tmpOthername: []int64{dnstime}},
				}
			}
		} else { //othername 不存在
			c := make(map[string]*othernameinfo)
			c[data.Dnstype] = &othernameinfo{
				sumquery: 1,
				Value:    map[string][]int64{tmpOthername: []int64{dnstime}},
			}
			targetmap[data.Value][tmpSecondname] = c
			// targetmap[tmpSecondname][tmpOthername] = c
		}
	} else { // secondname 不存在
		tmpmapothername := make(map[string]map[string]*othernameinfo) // othername : dnstype : othernameinfo
		c := make(map[string]*othernameinfo)                          // dnstype : othernameinfo
		c[data.Dnstype] = &othernameinfo{
			sumquery: 1,
			Value:    map[string][]int64{tmpOthername: []int64{dnstime}},
			// Value:    map[string][]int64{data.Value: []int64{dnstime}},
		}
		tmpmapothername[tmpSecondname] = c
		targetmap[data.Value] = tmpmapothername
	}
}

func findinsliceint64(needfind int64, targetslice []int64) bool {
	for _, tldsuffix := range targetslice {
		if needfind == tldsuffix {
			return true
		}
	}
	return false
}

func findinslicestr(needfind string, targetslice []string) bool {
	for _, tldsuffix := range targetslice {
		if needfind == tldsuffix {
			return true
		}
	}
	return false
}

func min(vals ...int) int {
	var min int
	for _, val := range vals {
		if min == 0 || val <= min {
			min = val
		}
	}
	return min
}

func max(vals ...int) int {
	var max int
	for _, val := range vals {
		if val > max {
			max = val
		}
	}
	return max
}
