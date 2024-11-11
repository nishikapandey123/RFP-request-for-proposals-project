// document.getElementById('upload-button').addEventListener('click', () => {
//     const fileInput = document.getElementById('file-input');
//     const file = fileInput.files[0];

//     if (!file) {
//         alert('Please select a file to upload.');
//         return;
//     }

//     const formData = new FormData();
//     formData.append('file', file);

//     fetch('/upload', {
//         method: 'POST',
//         body: formData,
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (data.error) {
//             alert(data.error);
//         } else {
//             document.getElementById('criteria-button').disabled = false;
//             document.getElementById('criteria-button').addEventListener('click', () => {
//                 displayQuestions(data.questions);
//             });
//         }
//     })
//     .catch(error => {
//         console.error('Error:', error);
//         alert('An error occurred while uploading the file.');
//     });
// });

// function displayQuestions(questions) {
//     const container = document.getElementById('questions-container');
//     const list = document.getElementById('questions-list');
//     list.innerHTML = '';

//     questions.forEach((question, index) => {
//         const card = document.createElement('div');
//         card.className = 'question-card';

//         const questionNumber = document.createElement('p');
//         questionNumber.textContent = `${index + 1}. ${question}`;
//         card.appendChild(questionNumber);

//         list.appendChild(card);
//     });

//     container.classList.remove('hidden');
//     window.scrollTo({ top: 0, behavior: 'smooth' });
// }
