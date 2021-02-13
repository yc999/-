package main

//{"timestamp":"1600993027","name":"0.205-240-81.adsl-dyn.isp.belgacom.be","type":"a",
//"value":"81.240.205.0"}
import (
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
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
type pDNS struct {
	tnow    string
	tbeg    string
	tend    string
	count   string
	rkey    string
	Dnstype string
	data    string
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
	path := "E:/wechatfile/WeChat Files/wxid_luhve56t0o4a11/FileStorage/File/2020-11/pdns_data"
	pkg := pDNS{}
	initglobal()
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
		pdnsUnmarshal(strT, &pkg)
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
	// fmt.Println(classifyMap)
	fmt.Println(uncalssifydata)
	fmt.Println(len(uncalssifydata))
	// fmt.Println(classifyMap)
	fmt.Println(classifysuccesstimes)
	path = "D:/dnswork/sharevm/unclassify.txt"
	err = filewrite(path, uncalssifydata)
	if err != nil {
		fmt.Println(err)
	}
}

//写入文件 path 写入文件路径  target 需要写入的内容
func filewrite(path string, target []string) error {
	// path = "D:/dnswork/sharevm/sortresult.txt"
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	// l, err := f.WriteString("Hello World")
	for _, k := range target {
		fmt.Fprintln(f, k)
	}
	if err != nil {
		f.Close()
		return err
	}
	fmt.Println("bytes written successfully")
	err = f.Close()
	if err != nil {
		return err
	}
	return nil
}

//解析数据
func pdnsUnmarshal(strT string, dnsdata *pDNS) {
	parts := strings.Split(strT, "\t")
	// parts = strings.Split(parts, " ")
	dnsdata.tnow = strings.Split(parts[0], ":")[1]
	dnsdata.tbeg = strings.Split(parts[1], ":")[1]
	dnsdata.tend = strings.Split(parts[2], ":")[1]
	dnsdata.count = strings.Split(parts[3], ":")[1]
	tmp := strings.Split(parts[4], ":")[1]
	tmp1 := strings.Split(tmp, "+")
	dnsdata.rkey = tmp1[0]
	dnsdata.Dnstype = tmp1[1]
	dnsdata.data = strings.Split(parts[5], ":")[1]
}

//用于统计前访问次数最多的二级域名
type sortname struct {
	name  string //二级域名
	count int    //访问次数
}

var sortnum int = 300
var sortnameSlice [300]sortname

var maxquerytimes int
var minquerytimes int = 0

//记录总共
var sortnamecount int = 0

//分类Map	类型 ：域名
var classifyMap map[string][]string

//已分类数据 域名 ： 类型
var classifydata map[string]string
var uncalssifydata []string

// 初始化参数
func initglobal() {
	classifyMap = make(map[string][]string)
	classifydata = make(map[string]string)
	shortnameMap = make(map[string]string)
	similarName = make(map[string][]string)
	mapSumquery = make(map[int64]*sumquerySecond)
	secondNameMap = make(map[string]map[string]*othernameinfo)
	secondNameMapShort = make(map[string]map[string]*othernameinfo)
	cdn2nameMap = make(map[string][]string)
	cctldpath := "D:/dnswork/sharevm/suffix"
	getcctld(cctldpath)
	datapath := "D:/dnswork/sharevm/hao.66360.cn.txt"
	getclassifydata(datapath)
	datapath = "D:/dnswork/sharevm/123.sogou.com.txt"
	getclassifydata(datapath)
	datapath = "D:/dnswork/sharevm/hao123.txt"
	getclassifydata(datapath)
	datapath = "D:/dnswork/sharevm/hao.360.com.txt"
	getclassifydata(datapath)
	path := "D:/dnswork/sharevm/alexclassify/"
	fs, _ := ioutil.ReadDir(path)
	for _, file := range fs {
		// fmt.Print(file.Name())
		getclassifydata(path + file.Name())
	}
	path = "D:/dnswork/sharevm/topchinaz/"
	fs, _ = ioutil.ReadDir(path)
	for _, file := range fs {
		fmt.Print(file.Name())
		getclassifydata(path + file.Name())
	}
	// fmt.Print(classifydata)
}

func getclassifydata(path string) {
	f, err := os.Open(path)
	checkError(err)
	defer f.Close()
	buf := bufio.NewReader(f)
	for {
		line, err := buf.ReadString('\n')
		parts := strings.Split(line, ",")
		// fmt.Println(parts)
		if len(parts) >= 2 {
			nameclass := parts[0]
			name := strings.TrimRight(parts[1], "\r\n")
			nameparts := strings.Split(name, "//")
			if len(nameparts) == 2 {
				name = nameparts[1]
			}
			nameparts = strings.Split(name, "www.")
			if len(nameparts) == 2 {
				name = nameparts[1]
			}
			nameparts = strings.Split(name, "/")
			if len(nameparts) >= 2 {
				name = nameparts[0]
			}
			_, have := classifydata[name]
			if !have {
				classifydata[name] = nameclass
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			panic(err)
		}
	}
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
		parts := strings.Split(line, ":")
		parts[0] = strings.TrimSpace(parts[0])
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

func solvedata(data *pDNS) {
	dnstime, err := strconv.ParseInt(data.tnow, 10, 64)
	dnstime = dnstime / 100000000
	if err != nil {
		fmt.Println("wrong message:")
		fmt.Println(data)
		return
	}
	// secondname, othername, nametype := parsename(data)
	// if nametype == 1 {
	// findsimlarname(secondname, othername, data.Name)
	// }
	// fmt.Println(data.Dnstype)
	if data.Dnstype == "1" || data.Dnstype == "28" {
		secondname, _, nametype := parsename(data)
		// findsimlarname(secondname, othername, data.Name)
		insert2sumquery(dnstime, data, secondname, nametype)
		classifyDomainname(data.rkey)
		// fmt.Println(data)
	} else if data.Dnstype == "cname" {
		// cnameanalyze(data)
	} else {
		// secondname, _, nametype := parsename(data)
		// // findsimlarname(secondname, othername, data.Name)
		// insert2sumquery(dnstime, data, secondname, nametype)
		// fmt.Println(data)
	}
	//insert2sumquery(dnstime, data, secondname, nametype)
	// if nametype == 1 {
	// 	collectsecondname(dnstime, data, secondname, othername, secondNameMapShort)
	// } else {
	// 	collectsecondname(dnstime, data, secondname, othername, secondNameMap)
	// }
	return
}

//分类
var classifysuccesstimes int = 0

func classifyDomainname(domainname string) {
	parts := strings.Split(domainname, ".")
	// tldname := parts[0]
	if parts[1] == "edu" || parts[0] == "edu" {
		classifysuccesstimes++
		add2classifymap("教育", domainname)
	} else if iftvclass(domainname) {

	} else {
		tmp2part := []string{parts[1], parts[0]}
		tmp2name := strings.Join(tmp2part, ".")

		if len(parts) >= 3 {
			tmp3part := []string{parts[2], parts[1], parts[0]}
			tmp3name := strings.Join(tmp3part, ".")
			if ifinstringMap(tmp3name, classifydata) {
				add2classifymap(classifydata[tmp3name], domainname)
				classifysuccesstimes++
				return
			} else if ifinstringMap(tmp2name, classifydata) {
				classifysuccesstimes++
				add2classifymap(classifydata[tmp2name], domainname)
				return
			}
		} else if ifinstringMap(tmp2name, classifydata) {
			classifysuccesstimes++
			add2classifymap(classifydata[tmp2name], domainname)
			return
		} else {
			if parts[0] == "cn" {
				// fmt.Println(tmp2name)
			}
			if !findinslicestr(tmp2name, uncalssifydata) {
				uncalssifydata = append(uncalssifydata, tmp2name)
			}
		}
	}
}

func iftvclass(domainname string) bool {
	// var mallist = []string {"tv", "video"}
	return false
}

//判断是否在Map[string]中
func ifinstringMap(data string, targetmap map[string]string) bool {
	_, have := targetmap[data]
	return have
}

func add2classifymap(index string, data string) {
	_, have := classifyMap[index]
	if have {
		if !findinslicestr(data, classifyMap[index]) {
			classifyMap[index] = append(classifyMap[index], data)

		}
	} else {
		classifyMap[index] = append(classifyMap[index], data)
	}
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
func insert2sumquery(dnstime int64, data *pDNS, secondname string, nametype int) {
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
func parsename(data *pDNS) (secondname string, othername string, nametype int) {
	var returentype int = 0
	parts := strings.Split(data.rkey, ".")
	partslen := len(parts)
	tldname := parts[0]

	var tmpSecondname, tmpOthername string
	if partslen >= 3 {
		if findinslicestr(tldname, counter.cctld[tldname[0:1]]) && partslen >= 3 {
			tmppart := []string{parts[2], parts[1], parts[0]}
			tmpSecondname = strings.Join(tmppart, ".")
			tmpOthername = strings.Join(parts[3:partslen], ".")
		} else {
			tmppart := []string{parts[1], parts[0]}
			tmpSecondname = strings.Join(tmppart, ".")
			tmpOthername = strings.Join(parts[2:partslen], ".")
		}
	} else {
		tmpSecondname = data.rkey
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
