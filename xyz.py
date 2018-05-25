def performAction( yp, rc, bc, action, drag, perform):

	if perform:
	 	cursor[0]= 4*(yp[0]-110)
		cursor[1]= 4*(yp[1]-120)
		
		if action == 'move':

			if yp[0]>110 and yp[0]<590 and yp[1]>120 and yp[1]<390:
				pyautogui.moveTo(cursor[0],cursor[1])
			elif yp[0]<110 and yp[1]>120 and yp[1]<390:
				pyautogui.moveTo( 8 , cursor[1])
			elif yp[0]>590 and yp[1]>120 and yp[1]<390:
				pyautogui.moveTo(1912, cursor[1])
			elif yp[0]>110 and yp[0]<590 and yp[1]<120:
				pyautogui.moveTo(cursor[0] , 8)
			elif yp[0]>110 and yp[0]<590 and yp[1]>390:
				pyautogui.moveTo(cursor[0] , 1072)
			elif yp[0]<110 and yp[1]<120:
				pyautogui.moveTo(8, 8)
			elif yp[0]<110 and yp[1]>390:
				pyautogui.moveTo(8, 1072)
			elif yp[0]>590 and yp[1]>390:
				pyautogui.moveTo(1912, 1072)
			else:
				pyautogui.moveTo(1912, 8)

		elif action == 'left':
			pyautogui.click(button = 'left')

		elif action == 'right':
			pyautogui.click(button = 'right')
			time.sleep(0.3)	

	elif action == 'up':
			pyautogui.scroll(5)
#			time.sleep(0.3)

		elif action == 'down':
			pyautogui.scroll(-5)			
#			time.sleep(0.3)

		elif action == 'drag' and drag == 'true':
			global y_pos
			drag = 'false'
			pyautogui.mouseDown()
		
			while(1):

				k = cv2.waitKey(10) & 0xFF
				changeStatus(k)

				_, frameinv = cap.read()
				# flip horizontaly to get mirror image in camera
				frame = cv2.flip( frameinv, 1)

				hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)

				b_mask = makeMask( hsv, blue_range)
				r_mask = makeMask( hsv, red_range)
				y_mask = makeMask( hsv, yellow_range)

				py_pos = y_pos 

				b_cen = drawCentroid( frame, b_area, b_mask, showCentroid)
				r_cen = drawCentroid( frame, r_area, r_mask, showCentroid)	
				y_cen = drawCentroid( frame, y_area, y_mask, showCentroid)
			
				if 	py_pos[0]!=-1 and y_cen[0]!=-1:
					y_pos = setCursorPos(y_cen, py_pos)

				performAction(y_pos, r_cen, b_cen, 'move', drag, perform)					
				cv2.imshow('Frame', frame)

				if distance(y_pos,r_cen)>60 or distance(y_pos,b_cen)>60 or distance(r_cen,b_cen)>60:
					break

			pyautogui.mouseUp()
