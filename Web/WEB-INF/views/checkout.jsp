<%@ page language="java" contentType="text/html; charset=utf-8"
	pageEncoding="utf-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%  String path = request.getContextPath();
	String basePath = request.getScheme()+"://"+request.getServerName()+":"+request.getServerPort()+path+"/";
%>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>Floral Shop, Check Out, Web Store</title>
<meta name="keywords"
	content="free template, floral shop, ecommerce, online shopping, store" />
<meta name="description"
	content="Floral Shop, Check Out, free template for ecommerce websites." />
<link href="<%=basePath%>templatemo_style.css" rel="stylesheet"
	type="text/css" />

<link rel="stylesheet" href="<%=basePath%>css/orman.css" type="text/css"
	media="screen" />
<link rel="stylesheet" href="<%=basePath%>css/nivo-slider.css"
	type="text/css" media="screen" />

<link rel="stylesheet" type="text/css"
	href="<%=basePath%>css/ddsmoothmenu.css" />
	<script src="http://code.jquery.com/jquery-1.11.1.min.js"></script>
<script type='text/javascript' src='<%=basePath%>js/logging.js'></script>
<script type="text/javascript" src="<%=basePath%>js/jquery.min.js"></script>
<script type="text/javascript" src="<%=basePath%>js/ddsmoothmenu.js">

/***********************************************
* Smooth Navigational Menu- (c) Dynamic Drive DHTML code library (www.dynamicdrive.com)
* This notice MUST stay intact for legal use
* Visit Dynamic Drive at http://www.dynamicdrive.com/ for full source code
***********************************************/

</script>

<script type="text/javascript">

ddsmoothmenu.init({
	mainmenuid: "templatemo_menu", //menu DIV id
	orientation: 'h', //Horizontal or vertical menu: Set to "h" or "v"
	classname: 'ddsmoothmenu', //class added to menu's outer DIV
	//customtheme: ["#1c5a80", "#18374a"],
	contentsource: "markup" //"markup" or ["container_id", "path_to_menu_file"]
})

function clearText(field)
{
    if (field.defaultValue == field.value) field.value = '';
    else if (field.value == '') field.value = field.defaultValue;
}

$(document).ready(function(){
$('#fullname').bind('input propertychange', function() {
    var name = $(this).val();
    
    if(!( /^[\u4E00-\u9FA5]{2,10}$/.test(name)) || name.length<2){ 
    	if (name ==""){
    		$("#nametip").css("color","#FEA5C5");
    		$('#nametip').html("&nbsp;&nbsp;&nbsp;??????????????????");
    	}else{

     		$("#nametip").css("color","#FEA5C5");
    		$('#nametip').html("&nbsp;&nbsp;&nbsp;????????????????????????2-10???");
    	}
    } 
    else{
     	$("#nametip").css("color","#FCE169");
    	$('#nametip').html("&nbsp;&nbsp;&nbsp;??????????????????");
    }
});


$('#address').bind('input propertychange', function() {
    var address = $(this).val();
    
    if(!( /^[\u4E00-\u9FA50-9]{10,50}$/.test(address)) || address.length<10){ 
    	if (address ==""){
    		$("#addresstip").css("color","#FEA5C5");
    		$('#addresstip').html("&nbsp;&nbsp;&nbsp;??????????????????");
    	}else{

     		$("#addresstip").css("color","#FEA5C5");
    		$('#addresstip').html("&nbsp;&nbsp;&nbsp;?????????????????????????????????10-50???");
    	}
    } 
    else{
     	$("#addresstip").css("color","#FCE169");
    	$('#addresstip').html("&nbsp;&nbsp;&nbsp;??????????????????");
    }
});


$('#city').bind('input propertychange', function() {
    var city = $(this).val();
    
    if(!( /^[\u4E00-\u9FA5]{2,10}$/.test(city)) || city.length<2){ 
    	if (city ==""){
    		$("#citytip").css("color","#FEA5C5");
    		$('#citytip').html("&nbsp;&nbsp;&nbsp;??????????????????");
    	}else{

     		$("#citytip").css("color","#FEA5C5");
    		$('#citytip').html("&nbsp;&nbsp;&nbsp;????????????????????????2-10???");
    	}
    } 
    else{
     	$("#citytip").css("color","#FCE169");
    	$('#citytip').html("&nbsp;&nbsp;&nbsp;??????????????????");
    }
});



$('#country').bind('input propertychange', function() {
    var country = $(this).val();
    
    if(!( /^[\u4E00-\u9FA5]{2,10}$/.test(country)) || country.length<2){ 
    	if (country ==""){
    		$("#countrytip").css("color","#FEA5C5");
    		$('#countrytip').html("&nbsp;&nbsp;&nbsp;??????????????????");
    	}else{

     		$("#countrytip").css("color","#FEA5C5");
    		$('#countrytip').html("&nbsp;&nbsp;&nbsp;????????????????????????2-10???");
    	}
    } 
    else{
     	$("#countrytip").css("color","#FCE169");
    	$('#countrytip').html("&nbsp;&nbsp;&nbsp;?????????????????????");
    }
});

$('#email').bind('input propertychange', function() {
    var email = $(this).val();
    
    if(!( /^[a-z0-9]+([._\\-]*[a-z0-9])*@([a-z0-9]+[-a-z0-9]*[a-z0-9]+.){1,63}[a-z0-9]+$/.test(email))){ 
    	if (email ==""){
    		$("#emailtip").css("color","#FEA5C5");
    		$('#emailtip').html("&nbsp;&nbsp;&nbsp;??????????????????");
    	}else{

     		$("#emailtip").css("color","#FEA5C5");
    		$('#emailtip').html("&nbsp;&nbsp;&nbsp;?????????????????????");
    	}
    } 
    else{
     	$("#emailtip").css("color","#FCE169");
    	$('#emailtip').html("&nbsp;&nbsp;&nbsp;??????????????????");
    }
});


$('#phone').bind('input propertychange', function() {
    var phone = $(this).val();
    
    if(!( /^1(3|4|5|7|8)\d{9}$/.test(phone)) ){ 
    	if (country ==""){
    		$("#teltip").css("color","#FEA5C5");
    		$('#teltip').html("&nbsp;&nbsp;&nbsp;??????????????????");
    	}else{

     		$("#teltip").css("color","#FEA5C5");
    		$('#teltip').html("&nbsp;&nbsp;&nbsp;???????????????????????????");
    	}
    } 
    else{
     	$("#teltip").css("color","#FCE169");
    	$('#teltip').html("&nbsp;&nbsp;&nbsp;??????????????????");
    }
});

});


$(function () {
	$("#checkoutbtn").click(function () {
		var namestatus =$('#nametip').text().replace(/\s+/g,"");
		//alert(namestatus);
		var addressstatus =$('#addresstip').text().replace(/\s+/g,""); 
		
		var citystatus = $('#citytip').text().replace(/\s+/g,"");
		var countrystatus = $('#countrytip').text().replace(/\s+/g,"");
		
		
		var emailstatus= $('#emailtip').text().replace(/\s+/g,"");
		var telstatus = $('#teltip').text().replace(/\s+/g,"");
	

		if (namestatus == "??????????????????" && addressstatus == "??????????????????" && citystatus == "??????????????????" && countrystatus == "?????????????????????"
				&& emailstatus =="??????????????????" && telstatus =="??????????????????"){
	  		$("#checkoutbtn").val("????????????");
	    	
	  		
	  		
	     	var name = $("input[name='fullname']").val();
	    	var address = $("input[name='address']").val();
	    	var city = $("input[name='city']").val();
	    	var country =  $("input[name='country']").val();
	    
	    	var email =  $("input[name='email']").val();
	    	var phone =  $("input[name='phone']").val();
	    	var allpay = $("#allpay").text().substring(1);
	    	
	    	
	 
	            		window.location.href="../shoppingAction/checkoutdeal.do?fullname="+name+"&address="+address+
						"&city="+city+"&country="+country+"&email="+email+"&phone="+phone+"&sumpay="+allpay;    	
	   
	            
	      
	    	
			
		}
		else{
			
		}
		
	});






})

</script>

<style>
#checkoutbtn{
    width: 60%;
    padding: 10px 0;
    font-size: 16px;
    font-weight: 100;
    background-color: transparent;
    color: red;
    font-weight: bold;
    margin-bottom: 10px;
    border: 1px solid rgba(238, 238, 238, 0.41);
    border-width: thin;
    cursor: pointer;
    outline: none;
    transition: 0.5s all;
    -webkit-transition: 0.5s all;
    -moz-transition: 0.5s all;
    -o-transition: 0.5s all;
    -ms-transition: 0.5s all;
    text-decoration: none;
    }
}

</style>
<style>
	.nav{
		color:#FBCF60;
		position:absolute;
		right:50px;
		top:10px;
		z-index:999;
		font-size: 20px;
		font-weight: bold;
		text-decoration: underline;
	}
</style>
</head>

<body>
	<c:choose>
		<c:when test="${not empty uid}">
			<div class="nav">??????,${username}</div>
		</c:when>
		<c:otherwise>
			<a class="nav" style="text-decoration: underline;" href="../jumpAction/login.do">?????????????????????!</a>
		</c:otherwise>
	</c:choose>
	<div id="templatemo_wrapper_sp">
		<div id="templatemo_header_wsp">
			<div id="templatemo_header" class="header_subpage">
				<div id="site_title">
					<a href="#" title=""></a>
				</div>
				<div id="templatemo_menu" class="ddsmoothmenu">
					<ul>
						<li><a href="../jumpAction/main.do" class="selected">??????</a></li>
						<li><a href="../jumpAction/login.do">????????????</a></li>
						<li><a href="../jumpAction/products.do">??????</a></li>
					<c:if test="${not empty uid}">
						<li><a href="../jumpAction/gotocart.do">?????????</a></li>
					</c:if>
						<li><a href="#">??????</a></li>
						<li><a href="../jumpAction/contact.do">????????????</a></li>
						<li><a href="../jumpAction/faqs.do">????????????</a></li>
							<c:if test="${not empty uid}">
						<li><a href="../jumpAction/userinfo.do">????????????</a></li>
				</c:if>
				<c:if test="${not empty uid}">
						<li><a href="../jumpAction/quit.do">????????????</a></li>
				</c:if>
					</ul>
					<div id="templatemo_search">
						<form action="../jumpAction/search.do" method="post">
							<input type="text"  name="keyword" required="required" id="keyword"
								title="keyword" onfocus="clearText(this)"
								onblur="clearText(this)" class="txt_field"  /> <input
								type="submit" name="" value="" alt="Search"
								id="searchbutton" title="Search" class="sub_btn" />
						</form>
					</div>
					<br style="clear: left" />
				</div>
				<a href="#" title="" class="site_repeat" target="_blank"><img
					src="<%=basePath%>images/top_repeat.png" title="" /></a>
				<!-- end of templatemo_menu -->

			</div>
			<!-- END of header -->
		</div>
		<!-- END of header wrapper -->
		<div id="templatemo_main_wrapper">
			<div id="templatemo_main">
				<div id="sidebar" class="left">
					<div class="sidebar_box">
						<span class="bottom"></span>
						<h3>??????</h3>
						<div class="content">
							<ul class="sidebar_list">
								<c:forEach items="${flowerkindlist}" var="flowerkind">
									<li><a href="#"><c:out value="${flowerkind.getName()}" /></a></li>
								</c:forEach>
							</ul>
						</div>
					</div>
					<div class="sidebar_box">
						<span class="bottom"></span>
						<h3>????????????</h3>
						<div class="content special">
							<img src="<%=basePath%>${bargainflower.getPicPath()}"
								alt="Flowers" /> <a href="#">${bargainflower.getName()}</a>
							<p>
								Price: <span class="price normal_price">${bargainflower.getPrice()}</span>&nbsp;&nbsp;
								<span class="price special_price">${bargainprice}</span>
							</p>
						</div>
					</div>
				</div>

				<div id="content" class="right">
					<h2>??????</h2>

					<h3>????????????</h3>
				<form >
					<div class="content_half left form_field">
					<div style="inline-block;"><p style="float: left;">??????</p>
					<div id="nametip" style="width:300px;color: red;font-size: 15px;" >&nbsp;&nbsp;&nbsp;???????????????</div>
						<input  name="fullname" type="text" id="fullname" maxlength="40"  />
						<br />
					</div>
					
					<div style="display:inline-block;">	
						<p style="float: left;">??????</p>
						<div id="addresstip" style="width:150px;color: red;font-size: 15px;" >&nbsp;&nbsp;&nbsp;???????????????</div>
						<input name="address" type="text" id="address" maxlength="40"  />
						
					</div>
					<div>
						<p style="float: left;">??????</p>
						<div id="citytip" style="width:150px;color: red;font-size: 15px;" >&nbsp;&nbsp;&nbsp;???????????????</div>
						<input  name="city" type="text" id="city" maxlength="40"  />
						
					</div>
					
					<div>
						<p style="float: left;">??????</p>
						<div id="countrytip" style="width:150px;color: red;font-size: 15px;" >&nbsp;&nbsp;&nbsp;???????????????</div>
						<input  name="country" type="text" id="country" maxlength="40"  />
					</div>	
					</div>

					<div class="content_half right form_field">
					<div>
						<p style="float: left;">????????????</p>
						<div id="emailtip" style="width:150px;color: red;font-size: 15px;" >&nbsp;&nbsp;&nbsp;???????????????</div>
						<input name="email" type="text" id="email" maxlength="40"  />
					</div>
					<div>
						<p style="float: left;">??????:</p>
						<div id="teltip" style="width:150px;color: red;font-size: 15px;" >&nbsp;&nbsp;&nbsp;???????????????</div>
						<input  name="phone" type="text" id="phone" maxlength="40"  /> 
					</div>
						<br />
						<span>?????????????????????????????????????????????????????????????????????</span>
					</div>

					<div class="cleaner h40"></div>

					<h3>?????????</h3>
					<div style="display: inline;">
					
						<div style="float: left;">??????: &nbsp;</div>
						<div style="" id="allpay">$${sumpay}</div>
					</div>
					<p>
				
				
					</p>
					<input id="checkoutbtn" class="submit" name="submit" type="button"   value="??????????????????"/>
			</form>
					<a href="#"><img src="<%=basePath%>images/free_shipping.jpg"
						alt="Free Shipping" /></a>
				</div>

			</div>

			<div class="cleaner"></div>
		</div>
		<!-- END of main -->
	</div>
	<!-- END of main wrapper -->




</body>
</html>