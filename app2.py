import os
import streamlit as st
import pandas as pd
import tempfile
import re

#from st_chat_message import message
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.models import MultiModal
from langchain_experimental.utilities import PythonREPL
from PIL import Image

class LPSolver:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        os.environ["LANGCHAIN_PROJECT"] = "Real World Challenge_LP SOLVER"

        # Initialize models and tools
        self.image_to_code_model = ChatOpenAI(temperature=0, model_name="gpt-4o")
        self.math_to_math_model = ChatOpenAI(temperature=0, model_name="gpt-4o")
        self.math_to_code_model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        self.python_tool = PythonREPL()

        # Initialize system and user prompts
        self.system_prompt = """
            LP 과목의 이해를 돕는 조교야. 주어진 LP 문제 이미지 혹은 문제의 상황을 형식에 맞게 글로 다시 적어줘.
            형식에 대한 예시도 줄게. 예시처럼 적어줘.

            예시 :
            maximize 3x_1 + 2x_2
            subject to
            x_1 + 2x_2 <= 4
            x_1 + x_2 <= 2
            x_1, x_2 >= 0
        """
        self.user_prompt = """
            다음의 문제상황 혹은 LP 문제 이미지를 해당하는 형식에 맞게 LP Formulation 해줘. 
            분, 시간 단위에 유의해서 분으로 단위를 통일해서 적어줘.
            금액은 만원이 아니라 원 단위로 적어줘.
            LaTeX 코드 형식으로 말고 일반 텍스트 형식으로 출력해줘.
        """

    def extract_math_problem(self, image_path: str) -> str:
        multimodal_model = MultiModal(
            self.image_to_code_model,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt
        )
        return multimodal_model.invoke(image_path)
    
    def check_math_again(self, math_problem: str) -> str:
        prompt = PromptTemplate.from_template(
            """
            주어진 {math}를 standard form으로 다시 작성해줘.
            괄호는 모두 벗겨서 수식으로 적고, 조건식은 아래와 같이 좌변에는 변수들만 남게 하고 우변에는 상수만 남겨줘.
            ex. 4y_1 - y_2 <= 200 
            조건식의 부등호는 <=이도록 식을 정리해줘. LaTeX를 사용하지 말고 적어줘.

            최종 출력값은 미사여구를 다 제외하고 수식만 적어줘.
            """
        )
        chain = prompt | self.math_to_math_model | StrOutputParser()
        return chain.invoke({"math": math_problem})

    def generate_scipy_code(self, problem_description: str) -> str:
        prompt = PromptTemplate.from_template(
            """
            {problem}을 SciPy 라이브러리의 optimize 모듈의 linprog 함수를 사용해서 풀 수 있도록 코드를 만들어.
            문제를 풀기 전에 목점함수가 min문제면 i=0, max문제면 i=1로 값을 할당해줘.
            또 제약 부등식을 코드에 넣기 전에 상수가 있는 우변이 더 크게 가르키도록 변형하고 코드에 넣어줘. 
            그리고 마지막 결과값을 출력할 때 i==0이면 그냥 출력하고, i==1이면 Objective value에 -1을 곱해서 출력해.

            제약 부등식을 부등호의 방향을 기준으로 먼저 구분하여 주석을 작성해.
            그리고 부등호의 방향이 >= 인 것이 있으면 양 변에 -1 을 곱해서 식을 재구성하여
            수정된 제약식을 다시 주석으로 작성해.
            그 뒤에 그 수정된 제약식을 바탕으로 matrix를 알맞게 만들어줘.

            그리고 Optimal Solution도 출력해줘.
            바로 실행할 수 있도록 코드만 적어줘.

            결과 출력은 아래의 형태로 해줘.

            If the solution is infeasible, then print out with 'print("It is infeasible")'.
            If the solution is unbounded, then print out with 'print("It is unbounded")'.
            """
        )
        chain = prompt | self.math_to_code_model | StrOutputParser()
        return chain.invoke({"problem": problem_description})

    def execute_code(self, code: str) -> str:
        try:
            return self.python_tool.run(code)
        except Exception as e:
            return f"Failed to execute.\nCode: {code}\nError: {type(e).__name__} - {e}"
        
    def check_ans(self, answer: str) -> int:
        # 정규 표현식을 사용해 'Objective value' 뒤의 숫자 추출
        match = re.search(r"Objective value: (\d+\.?\d*)", answer)
        
        if match:
            # 추출한 숫자를 반환 (소수점을 포함한 숫자도 처리)
            return int(float(match.group(1)))  # 소수점이 있을 수 있으므로 float로 변환 후 int로 변환
        else:
            raise ValueError("Objective value not found in the answer.")
        
    def solve_1(self, image_path: str) -> str:
        # Step 1: Extract math problem from image
        math_problem = self.extract_math_problem(image_path)

        # Step 2:  Check the problem again to suit the code
        check_math_again = self.check_math_again(math_problem)
        print("Standard Form:\n", check_math_again)
        print("----------------------------------------------")

        return check_math_again

    def solve_2(self, check_math_again: str) -> str:
        # Step 3: Generate Python code for solving the problem
        scipy_code = self.generate_scipy_code(check_math_again)
        print("Generated SciPy Code:\n", scipy_code)
        print("----------------------------------------------")

        # Step 3: Execute the generated code
        result = self.execute_code(scipy_code)
        print("Execution Result:", result)
        print("----------------------------------------------")

        return result
    

    def remove_latex(self, response):
        # LaTeX 포맷의 부분을 제거하거나 단순화
        cleaned_response = re.sub(r'\$.*?\$', '', response)  # $...$ 형태 제거
        return cleaned_response
    
        
    def main(self):
        # 가운데 정렬
        st.markdown("<h2 style='text-align: center;'>RWC: LP Solver</h2>", unsafe_allow_html=True)
        st.write("")
        st.write("""
                This model is made for students who study 'Linear Programming'. 
                We can solve all kinds of LP problems like feasible, infeasible, unbounded and so on. 
                I'll give you a simple guideline to help your better use.

                1. Upload your LP problem that you cannot solve.
                2. Check the formulation and compare with yours.
                3. Modify your answer and practice. 
                
                """)
        st.markdown("---")

        
        # 세션 상태 관리
        if "standard_form" not in st.session_state:
            st.session_state.standard_form = None
        if "answer" not in st.session_state:
            st.session_state.answer = None
        if "modeling_completed" not in st.session_state:
            st.session_state.modeling_completed = False

        uploaded_file = st.file_uploader("PNG 파일을 업로드하세요", type="png")

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image Problem.", use_container_width=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                image.save(tmp_file)
                tmp_file_path = tmp_file.name

            # 모델링 시작 버튼
            if st.button("모델링 시작"):
                with st.spinner("문제상황을 LP모델링하는 중입니다..."):
                    st.session_state.standard_form = solver.solve_1(tmp_file_path)
                    st.session_state.standard =solver.remove_latex(st.session_state.standard_form)
                    st.session_state.answer = solver.solve_2(st.session_state.standard_form)
                    st.session_state.value = solver.check_ans(st.session_state.answer)
                    st.session_state.modeling_completed = True  # 모델링 완료 상태 업데이트
                st.success("모델링이 완료되었습니다!")

        # 모델링 완료 후 버튼 표시
        if st.session_state.modeling_completed:
            if st.button("Standard Form 확인하기"):
                st.markdown(f"""
                    <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: #f0f0f0;">
                        <h4 style="color: #333;">Standard Form:</h4>
                        <div style="color: #333; font-family: Arial, sans-serif;">{st.session_state.standard}</div>
                    </div>
                """, unsafe_allow_html=True)

            user_answer = st.text_area("나의 답을 입력하세요:", height=100)

            if st.button("Answer 확인하기"):
                if user_answer.strip():  # 사용자가 답을 입력한 경우
                    user_int = int(user_answer.strip())
                    if user_int == st.session_state.value:
                        st.success("정답입니다! 🎉")
                    else:
                        st.error("틀린 답입니다. 다시 시도하세요.")
                else:
                    st.warning("답을 입력해 주세요.")  # 답이 비어 있는 경우

            if st.button("Answer 바로 확인하기"):
                st.markdown(f"""
                    <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: #f9f9f9;">
                        <h4 style="color: #333;">Answer:</h4>
                        <p style="color: #555;">{st.session_state.answer}</p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: #f9f9f9;">
                        <h4 style="color: #333;">Answer:</h4>
                        <p style="color: #555;">{st.session_state.value}</p>
                    </div>
                """, unsafe_allow_html=True)
                

if __name__ == "__main__":
    solver = LPSolver()
    solver.main()